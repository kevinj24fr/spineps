"""Quantitative measurements derived from SPINEPS segmentations.

These modules turn the masks the pipeline already produces into numbers. They add no new model and no new
inference: everything here is geometry and bookkeeping on existing output, which also means every number is
limited by the segmentation it came from and by the acquisition it was measured in.

Nothing in this package is clinically validated. Treat the outputs as measurements to check against a
reader, not as findings.
"""

from spineps.metrics.canal import CanalLevelMetrics, measure_canal
from spineps.metrics.normative import (
    SPINEPS_CANAL_OFFSET_MM,
    LevelReference,
    calibrate,
    is_unusually_narrow,
    load_reference,
    percentile_rank,
)
from spineps.metrics.numbering import NumberingConfidence, assess_numbering
from spineps.metrics.soft_tissue import SoftTissueVolumes, measure_soft_tissue

__all__ = [
    "SPINEPS_CANAL_OFFSET_MM",
    "CanalLevelMetrics",
    "LevelReference",
    "NumberingConfidence",
    "SoftTissueVolumes",
    "assess_numbering",
    "calibrate",
    "is_unusually_narrow",
    "load_reference",
    "measure_canal",
    "measure_soft_tissue",
    "percentile_rank",
]
