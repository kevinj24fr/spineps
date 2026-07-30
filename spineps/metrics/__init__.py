"""Quantitative measurements derived from SPINEPS segmentations.

These modules turn the masks the pipeline already produces into numbers. They add no new model and no new
inference: everything here is geometry and bookkeeping on existing output, which also means every number is
limited by the segmentation it came from and by the acquisition it was measured in.

Nothing in this package is clinically validated. Treat the outputs as measurements to check against a
reader, not as findings.
"""

from spineps.metrics.canal import CanalLevelMetrics, measure_canal
from spineps.metrics.numbering import NumberingConfidence, assess_numbering
from spineps.metrics.soft_tissue import SoftTissueVolumes, measure_soft_tissue

__all__ = [
    "CanalLevelMetrics",
    "NumberingConfidence",
    "SoftTissueVolumes",
    "assess_numbering",
    "measure_canal",
    "measure_soft_tissue",
]
