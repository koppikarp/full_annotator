"""Configuration dataclasses and YAML serialization for popcrop."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PreprocessingConfig:
    camera_offset: float = 0.0
    do_field_flattening: bool = False
    field_flattening_mode: str = "retrospective"  # "retrospective" or "reference"
    field_flattening_per_cycle: bool = True
    field_flattening_per_well: bool = True
    reference_flatfield_path: Optional[str] = None
    registration_method: str = "none"  # "none" or "phase_cross_correlation"
    registration_channel: int = 0  # channel used for registration (typically DAPI=0)
    chromatic_registration: bool = False
    chromatic_shift_red_green: List[float] = field(default_factory=lambda: [0.0, 0.0])
    psf_matching: bool = False
    psf_target_sigma: float = 1.0
    saturation_threshold: float = 65520.0
    unmixing_matrix: Optional[List[List[float]]] = None  # Nonnegative unmixing matrix (n_channels x n_channels)


@dataclass
class BlobDetectionConfig:
    bg_filter_size: int = 35
    bg_quantile: float = 0.15
    noise_estimator: str = "mad"  # "mad" or "std"
    top_k_cycles: int = 3
    persistence_threshold: float = 2.5  # z-score threshold to count cycle as active
    hysteresis_high: float = 4.0        # seed z-score / metric threshold
    hysteresis_low: float = 2.0         # expansion metric threshold
    min_blob_area: int = 8              # minimum blob size in pixels
    max_blob_fraction: float = 0.95     # maximum fraction of nucleus occupied


@dataclass
class BlobClassifierConfig:
    model_path: Optional[str] = None  # Optional legacy model; production uses manual_analysis
    prob_threshold: float = 0.70
    use_heuristic_fallback: bool = True
    classifier_type: str = "logistic_regression"  # "logistic_regression" or "gradient_boosting"


@dataclass
class ColorCallingConfig:
    core_distance_percentile: float = 0.35  # fraction of max distance defining core
    min_core_pixels: int = 4
    min_connected_yellow_core: int = 4      # min connected pixels required for yellow core call
    yellow_core_min_ratio: float = 0.20     # min ratio of Red/(Red+Green) in core to allow yellow
    green_rim_override_protection: bool = True
    max_spatial_separation: float = 4.0     # max distance (pixels) between disjoint R & G components
    min_colocalization_overlap: float = 0.25 # IoU overlap of R and G to permit yellow
    calibrated_color_model_path: Optional[str] = None  # Optional legacy model


@dataclass
class ManualAnalysisConfig:
    nuclear_diameter: float = 55.0
    radius_scale: float = 1.6
    batch_size: int = 64
    signal_probability_threshold: float = 0.70
    base_model_path: str = str(PROJECT_ROOT / "models/manual_parity/base_caller.joblib")
    blob_model_path: Optional[str] = str(PROJECT_ROOT / "models/manual_parity/blob_classifier.joblib")
    detector_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecodingConfig:
    barcode_column: str = "barcode_new"
    codebook_path: Optional[str] = str(PROJECT_ROOT / "data/library/v4.3_library_barcodes.csv")
    min_score_margin: float = 0.8           # margin between 1st and 2nd best barcode score
    min_confidence: float = 0.60
    allow_recombination: bool = False
    max_hamming_distance: int = 2
    match_reverse_complement: bool = True   # in situ sequencing reads RC of library template


@dataclass
class OutputConfig:
    output_dir: str = "popcrop_output"
    save_probability_maps: bool = True
    save_masks: bool = True
    save_diagnostics: bool = True
    diagnostic_max_cells: int = 50
    export_format: str = "parquet"          # "parquet" or "csv"


@dataclass
class PipelineConfig:
    analysis_mode: str = "manual"  # shared raw annotator path, or explicit legacy
    plate_id: str = "Plate1"
    well_id: str = "WellA1"
    tile_id: str = "Tile1"
    experiment_date: Optional[str] = None  # Explicit acquisition/experiment date, YYYY-MM-DD
    random_seed: int = 42
    channel_names: List[str] = field(default_factory=lambda: ["Red", "Green"])
    has_dapi_cycle: bool = True             # True if cycle 0 is DAPI/phenotype
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    blob_detection: BlobDetectionConfig = field(default_factory=BlobDetectionConfig)
    blob_classifier: BlobClassifierConfig = field(default_factory=BlobClassifierConfig)
    color_calling: ColorCallingConfig = field(default_factory=ColorCallingConfig)
    manual_analysis: ManualAnalysisConfig = field(default_factory=ManualAnalysisConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self, file_path: str | Path) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, file_path: str | Path) -> PipelineConfig:
        path = Path(file_path)
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        # Reconstruct nested dataclasses
        prep_data = data.pop("preprocessing", {})
        blob_data = data.pop("blob_detection", {})
        clf_data = data.pop("blob_classifier", {})
        col_data = data.pop("color_calling", {})
        dec_data = data.pop("decoding", {})
        out_data = data.pop("output", {})
        manual_data = data.pop("manual_analysis", {})

        return cls(
            preprocessing=PreprocessingConfig(**prep_data),
            blob_detection=BlobDetectionConfig(**blob_data),
            blob_classifier=BlobClassifierConfig(**clf_data),
            color_calling=ColorCallingConfig(**col_data),
            decoding=DecodingConfig(**dec_data),
            output=OutputConfig(**out_data),
            manual_analysis=ManualAnalysisConfig(**manual_data),
            **data
        )
