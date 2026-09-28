"""File-run integration: concurrent jobs, provenance and complete exports."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from popcrop.config import PipelineConfig
from popcrop.io import run_identity
from popcrop.pipeline import OpsPipeline


def test_concurrent_runs_get_distinct_directories(tmp_path):
    with ThreadPoolExecutor(max_workers=4) as pool:
        runs = list(pool.map(lambda _: run_identity.allocate_run(tmp_path, 'image.tif', PipelineConfig()), range(12)))
    assert len({directory for directory, _, _ in runs}) == 12
    assert all(directory.is_dir() for directory, _, _ in runs)


def test_atomic_collision_preserves_existing_output(tmp_path, monkeypatch):
    class FixedTime:
        @staticmethod
        def now(tz):
            return datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(run_identity, 'datetime', FixedTime)
    ids = iter(['a'*32, 'a'*32, 'b'*32])
    monkeypatch.setattr(run_identity, 'uuid4', lambda: SimpleNamespace(hex=next(ids)))
    first, _, _ = run_identity.allocate_run(tmp_path, 'image.tif', PipelineConfig())
    sentinel = first/'existing-result.txt'
    sentinel.write_text('original')
    second, _, _ = run_identity.allocate_run(tmp_path, 'image.tif', PipelineConfig())
    assert first != second and sentinel.read_text() == 'original'


def test_safe_bounded_names_keep_original_identity(tmp_path):
    cfg = PipelineConfig(plate_id='../plate/'+'x'*200, well_id='B/2', tile_id='../../tile', experiment_date='2026-09-28')
    source = tmp_path/('long image '*20+'.tif')
    directory, prefix, metadata = run_identity.allocate_run(tmp_path/'out', source, cfg)
    assert directory.parent == tmp_path/'out'
    assert '/' not in prefix and len(prefix+'_nucleus_barcode_calls.csv') <= 255
    assert metadata['source_image'] == source.name and metadata['plate'] == cfg.plate_id
    assert metadata['run_started_utc'].endswith('Z') and 'date-2026-09-28' in prefix


def _file_pipeline(monkeypatch, cfg):
    """Exercise real exports without rerunning scientific image analysis."""
    import popcrop.pipeline as module
    obj = OpsPipeline.__new__(OpsPipeline)
    obj.config = cfg
    mask = np.ones((4, 4), dtype=np.int32)
    monkeypatch.setattr(module, 'load_raw_image', lambda **kwargs: (
        np.zeros((1, 2, 4, 4), dtype=np.uint16), mask, {'source_path': str(kwargs['image_path_or_paths'])}))
    monkeypatch.setattr(module, 'load_or_generate_nuclear_mask', lambda **kwargs: mask)
    def analyze(**kwargs):
        nuclei = pd.DataFrame([dict(nucleus_id=1, has_detectable_spot=True, has_signal=True,
            decoding_status='matched', nearest_hamming_distance=0, hamming_distance=0,
            best_gene='GENE1', assigned_gene='GENE1', assigned_barcode='ACGT',
            gene_candidates=['GENE1'], nearest_barcodes=['ACGT'], nearest_genes=['GENE1'])])
        obj.df_cycles = pd.DataFrame([dict(nucleus_id=1, cycle=1, called_base='A')])
        return nuclei, nuclei.copy(), {'nuclear_mask': mask, 'blob_mask': mask,
                                       'blob_prob_map': np.ones((4, 4), dtype=np.float32)}
    monkeypatch.setattr(obj, 'run_on_arrays', analyze)
    monkeypatch.setattr(module, 'generate_diagnostic_overlays',
                        lambda **kwargs: Path(kwargs['output_path']).write_bytes(b'test-diagnostic'))
    return obj


def test_exports_share_identity_and_reruns_never_overwrite(tmp_path, monkeypatch):
    cfg = PipelineConfig(plate_id='P2', well_id='C4', tile_id='009', experiment_date='2026-09-28')
    cfg.decoding.codebook_path = None
    obj = _file_pipeline(monkeypatch, cfg)
    image = tmp_path/'wellc4_009_RGB.tif'
    first = obj.run_on_files(image_path=image, output_dir=tmp_path/'results')
    original_csv = Path(first['saved_files']['barcode_calls_csv']).read_bytes()
    second = obj.run_on_files(image_path=image, output_dir=tmp_path/'results')
    assert first['output_dir'] != second['output_dir']
    assert Path(first['saved_files']['barcode_calls_csv']).read_bytes() == original_csv
    assert first['df_nuclei'].nucleus_uid[0] != second['df_nuclei'].nucleus_uid[0]
    metadata = first['run_metadata']
    for path in first['saved_files'].values():
        assert Path(path).exists() and Path(path).name.startswith(metadata['filename_prefix']+'_')
    for key in ['nuclei', 'components', 'cycles', 'barcode_calls_csv', 'gene_abundance']:
        path = first['saved_files'][key]
        frame = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path, dtype={'tile':str})
        assert frame.source_image.tolist() == [image.name] and frame.run_id.tolist() == [metadata['run_id']]
        assert frame.well.tolist() == ['C4'] and frame.tile.tolist() == ['009']
        assert frame.experiment_date.tolist() == ['2026-09-28']
        if key != 'gene_abundance':
            assert frame.nucleus_uid.tolist() == [metadata['run_id']+':1']
    report = json.loads(Path(first['saved_files']['summary_json']).read_text())
    assert report['run'] == metadata and report['runtime_seconds'] >= 0
    manifest = json.loads(Path(first['saved_files']['manifest']).read_text())
    assert manifest['status'] == 'complete' and manifest['saved_files'] == first['saved_files']


def test_failed_export_does_not_claim_completion(tmp_path, monkeypatch):
    import popcrop.pipeline as module
    cfg = PipelineConfig()
    cfg.decoding.codebook_path = None
    obj = _file_pipeline(monkeypatch, cfg)
    def fail(**kwargs):
        raise RuntimeError('Export interrupted')
    monkeypatch.setattr(module, 'generate_diagnostic_overlays', fail)
    with pytest.raises(RuntimeError, match='Export interrupted'):
        obj.run_on_files(image_path=tmp_path/'image.tif', output_dir=tmp_path/'results')
    manifest_path, = (tmp_path/'results').glob('*/*_run_manifest.json')
    assert json.loads(manifest_path.read_text())['status'] == 'started'
