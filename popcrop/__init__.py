"""
POPCROP: Production-Quality Pipeline for Detecting and Base-Calling
Extended Fluorescent Signals in Optical Pooled Screening.
"""

__version__ = "0.1.0"

from popcrop.pipeline import OpsPipeline
from popcrop.config import PipelineConfig

__all__ = ["OpsPipeline", "PipelineConfig"]
