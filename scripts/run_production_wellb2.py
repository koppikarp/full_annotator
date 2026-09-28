"""Run raw annotator-compatible analysis on this assay's multicycle images."""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import sys
import os
import tempfile
import warnings

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'popcrop-matplotlib'))
warnings.filterwarnings('ignore', category=FutureWarning, message='.*deprecated.*')

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_IMAGE = ROOT/'wellb2_002_002_RGB.tif'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from popcrop.config import PipelineConfig
from popcrop.pipeline import OpsPipeline
from popcrop.io.run_identity import normalize_experiment_date


def experiment_date_argument(value):
    try:
        return normalize_experiment_date(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=REFERENCE_IMAGE)
    parser.add_argument('--codebook', type=Path)
    parser.add_argument('--barcode-column')
    parser.add_argument('--out', type=Path, help='Results parent directory; each run creates a unique image/date-named child')
    parser.add_argument('--diameter', type=float)
    parser.add_argument('--mask', type=Path, help='Optional precomputed integer nuclear mask')
    parser.add_argument('--config', type=Path, help='Optional YAML settings; explicit CLI options take precedence')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--diagnostic-cells', type=int)
    parser.add_argument('--plate', help='Optional plate identity')
    parser.add_argument('--well', help='Optional well identity')
    parser.add_argument('--tile', help='Optional tile identity; otherwise input filename for new images')
    parser.add_argument('--experiment-date', '--date', type=experiment_date_argument,
                        help='Optional acquisition/experiment date, YYYY-MM-DD; separate from automatic UTC run time')
    return parser


def build_run_config(args):
    cfg = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()
    cfg.analysis_mode = 'manual'
    is_reference = args.image.resolve() == REFERENCE_IMAGE.resolve()
    if not args.config:
        cfg.plate_id = 'Plate1' if is_reference else 'unspecified'
        cfg.well_id = 'B2' if is_reference else 'unspecified'
        cfg.tile_id = '14' if is_reference else args.image.stem
        cfg.output.diagnostic_max_cells = 12
    for argument, attribute in [('plate','plate_id'),('well','well_id'),('tile','tile_id')]:
        if getattr(args, argument) is not None:
            setattr(cfg, attribute, getattr(args, argument))
    if args.experiment_date is not None:
        cfg.experiment_date = args.experiment_date
    cfg.experiment_date = normalize_experiment_date(cfg.experiment_date)
    if args.diameter is not None:
        cfg.manual_analysis.nuclear_diameter = args.diameter
    if args.batch_size is not None:
        cfg.manual_analysis.batch_size = args.batch_size
    if args.codebook is not None:
        cfg.decoding.codebook_path = str(args.codebook.resolve())
    if args.barcode_column is not None:
        cfg.decoding.barcode_column = args.barcode_column
    if args.diagnostic_cells is not None:
        cfg.output.diagnostic_max_cells = args.diagnostic_cells
        cfg.output.save_diagnostics = args.diagnostic_cells > 0
    if args.out is not None:
        output_dir = args.out
    elif args.config:
        output_dir = Path(cfg.output.output_dir)
    else:
        output_dir = ROOT/'popcrop_output'
    cfg.output.output_dir = str(output_dir.resolve())
    return cfg, output_dir


def main():
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    cfg, output_dir = build_run_config(args)
    result = OpsPipeline(cfg).run_on_files(image_path=args.image, mask_path=args.mask,
                                          codebook_path=cfg.decoding.codebook_path, output_dir=output_dir)
    report_path = Path(result['saved_files']['summary_json'])
    summary = json.loads(report_path.read_text())
    print(Path(result['saved_files']['summary_report']).read_text())
    print(f"Runtime: {summary['runtime_seconds']:.1f} seconds. Results: {result['output_dir']}")

if __name__ == '__main__':
    main()
