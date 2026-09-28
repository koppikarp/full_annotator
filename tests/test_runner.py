"""Prevent new image runs from inheriting the reference image's identity/settings."""
import pytest
from scripts.run_production_wellb2 import ROOT, build_parser, build_run_config
from popcrop.config import PipelineConfig


def test_new_image_has_separate_output_and_no_invented_plate_identity(tmp_path):
    args = build_parser().parse_args(['--image', str(tmp_path/'new_field.tif')])
    cfg, out = build_run_config(args)
    assert cfg.plate_id == cfg.well_id == 'unspecified'
    assert cfg.tile_id == 'new_field'
    assert out == ROOT/'popcrop_output'
    assert cfg.decoding.barcode_column == 'barcode_new'
    assert cfg.manual_analysis.nuclear_diameter == 55


def test_yaml_settings_survive_unset_cli_defaults(tmp_path):
    cfg = PipelineConfig(plate_id='P2', well_id='C4', tile_id='9')
    cfg.manual_analysis.batch_size = 8
    cfg.decoding.barcode_column = 'barcode'
    cfg.output.output_dir = str(tmp_path/'configured_output')
    path = tmp_path/'config.yaml'
    cfg.to_yaml(path)
    args = build_parser().parse_args(['--image',str(tmp_path/'other.tif'), '--config',str(path), '--tile','10'])
    actual, out = build_run_config(args)
    assert (actual.plate_id, actual.well_id, actual.tile_id) == ('P2','C4','10')
    assert actual.manual_analysis.batch_size == 8
    assert actual.decoding.barcode_column == 'barcode'
    assert out == tmp_path/'configured_output'


def test_experiment_date_cli_precedence(tmp_path):
    cfg = PipelineConfig(experiment_date='2026-09-01')
    path = tmp_path/'config.yaml'
    cfg.to_yaml(path)
    assert build_run_config(build_parser().parse_args(['--config', str(path)]))[0].experiment_date == '2026-09-01'
    args = build_parser().parse_args(['--config', str(path), '--date', '2026-09-28'])
    assert build_run_config(args)[0].experiment_date == '2026-09-28'


@pytest.mark.parametrize('value', ['2026-02-30', '09-28-2026', '20260928'])
def test_invalid_experiment_date_is_rejected(value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(['--experiment-date', value])
