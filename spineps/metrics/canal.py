"""Per-level spinal canal geometry, for quantifying canal narrowing from a SPINEPS segmentation.

What this measures and what it does not
---------------------------------------
SPINEPS segments the spinal canal on sagittal images. From that mask this module reports, for every
vertebral level and every intervertebral disc level it can identify:

* ``ap_diameter_mm`` -- the anterior-posterior canal diameter, the standard sagittal measure of canal
  narrowing, reported as the minimum, mean and median over the slices belonging to the level.
* ``canal_volume_mm3`` -- the canal volume within the level's cranio-caudal band.
* ``sagittal_area_mm2`` -- the canal's area in the mid-sagittal plane within that band.

``sagittal_area_mm2`` is deliberately *not* called a cross-sectional area. The dural sac cross-sectional
area used in axial stenosis grading is measured perpendicular to the canal axis on axial images; it cannot
be obtained from a sagittal acquisition and is not what this returns.

The AP diameter is the measure that is genuinely native to a sagittal acquisition, which is why it is the
primary output here. Its reliability is bounded by slice thickness: a typical sagittal T2w has ~3-4 mm
slices, so "the mid-sagittal slice" is a thick slab and a canal that is narrowest off-midline will be
under-measured. Published AP-diameter thresholds for stenosis exist, but they depend on acquisition and
measurement convention -- check them against current literature for your protocol rather than assuming the
numbers here are directly comparable.

Use the median, not the minimum
-------------------------------
Measured over 147 levels of a real cohort segmented on two different compute backends, the statistics are
not equally trustworthy:

===============  ==================  =================
statistic        mean abs difference  worst difference
===============  ==================  =================
mean_ap          0.13 mm             1.00 mm
median_ap        0.18 mm             1.21 mm
min_ap           0.33 mm             **13.56 mm**
===============  ==================  =================

``min_ap_diameter_mm`` is reported because it is the intuitive thing to ask for, but it is the least
reliable number here and should not be used as an endpoint on its own. Two reasons: a single-slice outlier
sets it, and at the caudal end the canal genuinely tapers, so on this cohort the "narrowest level" was S1
in most subjects -- anatomy, not stenosis. Prefer ``median_ap_diameter_mm``, and look at a mobile level you
chose deliberately rather than at whichever level happened to score lowest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import Optional

import numpy as np
from TPTBox import NII, Log_Type, No_Logger
from TPTBox.core.vert_constants import Location

from spineps.metrics.labels import decode_vertebra_label, vertebra_name

logger = No_Logger(prefix="CanalMetrics")

# Canonical orientation for the measurement, chosen so the axes are unambiguous:
# axis 0 = left->right (sagittal slice index), axis 1 = posterior->anterior, axis 2 = inferior->superior.
_MEASURE_ORIENTATION = ("R", "A", "S")
_AXIS_LR, _AXIS_PA, _AXIS_IS = 0, 1, 2

#: A level is only reported if at least this many canal voxels fall inside its band.
MIN_CANAL_VOXELS_PER_LEVEL = 10

#: Default floor on a row's AP extent before it counts towards a level's diameters. Rows thinner than this
#: are the taper at the ends of the canal or stray voxels; including them made the reported minimum
#: implausible (sub-millimetre "diameters") on real data. Expressed in mm so it does not silently change
#: meaning with voxel size, and exposed as a parameter because it is a judgement call, not a constant.
DEFAULT_MIN_AP_EXTENT_MM = 2.0


@dataclass
class CanalLevelMetrics:
    """Canal geometry for a single vertebral or disc level.

    Attributes:
        level_label (int): Instance label of the vertebra, or of the vertebra above a disc level.
        level_name (str): Human-readable level name, e.g. "L4" or "L4-L5".
        is_disc_level (bool): True if this row describes an intervertebral disc level.
        n_slices (int): Number of cranio-caudal slices the level spans.
        min_ap_diameter_mm (float): Narrowest AP canal diameter across the level's slices. Fragile -- see
            the module docstring; prefer the median.
        mean_ap_diameter_mm (float): Mean AP canal diameter across the level's slices.
        median_ap_diameter_mm (float): Median AP canal diameter across the level's slices.
        canal_volume_mm3 (float): Canal volume within the level's band.
        sagittal_area_mm2 (float): Canal area in the mid-sagittal plane within the level's band.
        canal_voxels (int): Number of canal voxels in the band, for judging whether the row is trustworthy.
    """

    level_label: int
    level_name: str
    is_disc_level: bool
    n_slices: int
    min_ap_diameter_mm: float
    mean_ap_diameter_mm: float
    median_ap_diameter_mm: float
    canal_volume_mm3: float
    sagittal_area_mm2: float
    canal_voxels: int

    def as_row(self) -> dict:
        """Flattens the metrics into a dict suitable for a TSV row.

        Returns:
            dict: One key per field, in declaration order.
        """
        return {
            "level_label": self.level_label,
            "level_name": self.level_name,
            "is_disc_level": self.is_disc_level,
            "n_slices": self.n_slices,
            "min_ap_diameter_mm": round(self.min_ap_diameter_mm, 3),
            "mean_ap_diameter_mm": round(self.mean_ap_diameter_mm, 3),
            "median_ap_diameter_mm": round(self.median_ap_diameter_mm, 3),
            "canal_volume_mm3": round(self.canal_volume_mm3, 3),
            "sagittal_area_mm2": round(self.sagittal_area_mm2, 3),
            "canal_voxels": self.canal_voxels,
        }


@dataclass
class CanalMeasurement:
    """All per-level canal measurements for one scan.

    Attributes:
        levels (list[CanalLevelMetrics]): One entry per measured vertebral or disc level.
        warnings (list[str]): Conditions that limit how far the numbers should be trusted.
        voxel_size_mm (tuple[float, float, float]): Voxel spacing the measurement was made at, in the
            measurement orientation (left-right, posterior-anterior, inferior-superior).
    """

    levels: list[CanalLevelMetrics] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    voxel_size_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def as_rows(self) -> list[dict]:
        """Returns every level as a dict row, narrowest level first by median AP diameter.

        Ordering uses the median rather than the minimum: across a real cohort the minimum differed by up
        to 13.6 mm between compute backends, so ordering by it would reshuffle the table for reasons that
        have nothing to do with the patient.

        Returns:
            list[dict]: Rows ordered by ascending median AP diameter.
        """
        return [lvl.as_row() for lvl in sorted(self.levels, key=lambda x: x.median_ap_diameter_mm)]


def _ap_diameters_mm(canal_midsagittal: np.ndarray, spacing_pa: float, min_extent_mm: float) -> np.ndarray:
    """Computes the AP canal diameter along a single mid-sagittal plane.

    The AP diameter is conventionally read off the mid-sagittal slice, so that is what this uses. Unioning
    the canal across every sagittal slice instead would report the widest AP extent found anywhere across
    the column's width, which on a curved canal overestimates the diameter at every level.

    For each cranio-caudal row the diameter is the span between the first and last canal voxel along the AP
    axis. A span rather than a voxel count means an interior hole from a segmentation artefact does not read
    as a narrower canal.

    Args:
        canal_midsagittal (np.ndarray): Boolean canal mask for one sagittal slice, in (PA, IS) axis order.
        spacing_pa (float): Voxel spacing along the AP axis in mm.
        min_extent_mm (float): Rows with less AP extent than this are ignored as taper or artefact.

    Returns:
        np.ndarray: AP diameter in mm for every row with enough canal, empty if none qualify.
    """
    diameters = []
    for is_index in range(canal_midsagittal.shape[1]):
        column = canal_midsagittal[:, is_index]
        if not column.any():
            continue
        occupied = np.flatnonzero(column)
        extent = (occupied[-1] - occupied[0] + 1) * spacing_pa
        if extent < min_extent_mm:
            continue
        diameters.append(extent)
    return np.asarray(diameters, dtype=float)


def _mid_sagittal_index(canal: np.ndarray) -> int:
    """Finds the sagittal slice index closest to the canal's centre of mass.

    Args:
        canal (np.ndarray): Boolean canal mask in (LR, PA, IS) axis order.

    Returns:
        int: Index along the left-right axis, or 0 if the mask is empty.
    """
    per_slice = canal.sum(axis=(1, 2))
    if per_slice.sum() == 0:
        return 0
    positions = np.arange(per_slice.size)
    return round(float((positions * per_slice).sum() / per_slice.sum()))


def _bands_from_mask(labelled: np.ndarray) -> dict[int, tuple[int, int]]:
    """Assigns each cranio-caudal slice to exactly one label, then returns each label's contiguous extent.

    Taking each label's full extent independently does not work on a real spine: the column is curved, so
    axis-aligned extents of adjacent vertebrae overlap, and a single narrow slice then lowers the reported
    minimum diameter of several levels at once. Instead every slice is given to whichever label occupies
    most of it, which makes the bands a partition.

    Args:
        labelled (np.ndarray): Integer mask in (LR, PA, IS) axis order.

    Returns:
        dict[int, tuple[int, int]]: Label to (first, last) inclusive index along the IS axis.
    """
    labels = [int(v) for v in np.unique(labelled) if v != 0]
    if not labels:
        return {}

    owner: dict[int, int] = {}
    for is_index in range(labelled.shape[_AXIS_IS]):
        plane = labelled[:, :, is_index]
        counts = {label: int((plane == label).sum()) for label in labels}
        best = max(counts, key=lambda lab: counts[lab])
        if counts[best] > 0:
            owner[is_index] = best

    bands: dict[int, tuple[int, int]] = {}
    for label in labels:
        owned = sorted(idx for idx, lab in owner.items() if lab == label)
        if owned:
            bands[label] = (owned[0], owned[-1])
    return bands


def measure_canal(
    semantic_nii: NII,
    vert_nii: NII,
    include_disc_levels: bool = True,
    min_ap_extent_mm: float = DEFAULT_MIN_AP_EXTENT_MM,
    logger_=None,
) -> CanalMeasurement:
    """Measures spinal canal geometry at every level of a SPINEPS segmentation.

    Args:
        semantic_nii (NII): Semantic mask containing the spinal canal (and optionally the discs).
        vert_nii (NII): Vertebra instance mask giving each level its own label.
        include_disc_levels (bool, optional): Also measure the gaps between consecutive vertebrae, which is
            where canal narrowing is usually graded. Defaults to True.
        min_ap_extent_mm (float, optional): Ignore mid-sagittal rows thinner than this when computing
            diameters, so canal taper and stray voxels do not set the reported minimum. Set to 0 to
            measure raw geometry. Defaults to DEFAULT_MIN_AP_EXTENT_MM.
        logger_ (optional): Logger for warnings. Defaults to None (module logger).

    Returns:
        CanalMeasurement: Per-level metrics plus any warnings about their reliability.

    Raises:
        ValueError: If the semantic mask contains no spinal canal label.
    """
    log = logger_ if logger_ is not None else logger

    semantic = semantic_nii.copy().reorient_(_MEASURE_ORIENTATION)
    verts = vert_nii.copy().reorient_(_MEASURE_ORIENTATION)

    canal_labels = [Location.Spinal_Canal.value, Location.Spinal_Cord.value]
    present = set(semantic.unique().tolist()) if hasattr(semantic.unique(), "tolist") else set(semantic.unique())
    if not present.intersection(canal_labels):
        raise ValueError(
            f"semantic mask contains no spinal canal label (looked for {canal_labels}, found {sorted(present)}). "
            "The canal is only produced by the semantic phase; check that it ran."
        )

    canal = semantic.extract_label(canal_labels).get_seg_array().astype(bool)
    vert_arr = verts.get_seg_array().astype(np.int32)
    spacing = tuple(float(z) for z in semantic.zoom)
    voxel_volume = spacing[0] * spacing[1] * spacing[2]

    result = CanalMeasurement(voxel_size_mm=spacing)  # type: ignore[arg-type]

    if canal.shape != vert_arr.shape:
        raise ValueError(f"semantic and vertebra masks have different shapes: {canal.shape} vs {vert_arr.shape}")

    # A thick sagittal acquisition cannot resolve an off-midline narrowing; say so once, up front.
    if spacing[_AXIS_LR] > 2.0:
        result.warnings.append(
            f"sagittal slice spacing is {spacing[_AXIS_LR]:.2f} mm; AP diameters are averaged over thick "
            "slices and a narrowing that is worst off-midline will be under-measured"
        )

    mid_sag = _mid_sagittal_index(canal)

    # The instance mask encodes a vertebra, its disc and its endplate as the same index offset by
    # multiples of 100. Split them apart, or every vertebra would be measured three times over bands
    # that overlap each other.
    vertebra_only = np.zeros_like(vert_arr)
    disc_only = np.zeros_like(vert_arr)
    for raw in (int(v) for v in np.unique(vert_arr) if v != 0):
        index, subregion = decode_vertebra_label(raw)
        if subregion == 0:
            vertebra_only[vert_arr == raw] = index
        elif subregion == 1:
            disc_only[vert_arr == raw] = index

    bands = _bands_from_mask(vertebra_only)
    disc_bands = _bands_from_mask(disc_only)

    def measure_band(label: int, name: str, lo: int, hi: int, is_disc: bool) -> Optional[CanalLevelMetrics]:
        sub = canal[:, :, lo : hi + 1]
        voxels = int(sub.sum())
        if voxels < MIN_CANAL_VOXELS_PER_LEVEL:
            return None
        mid_plane = sub[mid_sag] if mid_sag < sub.shape[0] else sub.any(axis=0)
        diameters = _ap_diameters_mm(mid_plane, spacing[_AXIS_PA], min_ap_extent_mm)
        if diameters.size == 0:
            return None
        return CanalLevelMetrics(
            level_label=label,
            level_name=name,
            is_disc_level=is_disc,
            n_slices=int(hi - lo + 1),
            min_ap_diameter_mm=float(diameters.min()),
            mean_ap_diameter_mm=float(diameters.mean()),
            median_ap_diameter_mm=float(np.median(diameters)),
            canal_volume_mm3=voxels * voxel_volume,
            sagittal_area_mm2=float(mid_plane.sum()) * spacing[_AXIS_PA] * spacing[_AXIS_IS],
            canal_voxels=voxels,
        )

    for label, (lo, hi) in sorted(bands.items()):
        row = measure_band(label, vertebra_name(label), lo, hi, is_disc=False)
        if row is not None:
            result.levels.append(row)

    if include_disc_levels:
        if disc_bands:
            # The mask labels discs explicitly; use them rather than inferring a gap.
            for index, (lo, hi) in sorted(disc_bands.items()):
                row = measure_band(index, f"{vertebra_name(index)}_disc", lo, hi, is_disc=True)
                if row is not None:
                    result.levels.append(row)
        else:
            # No disc labels present: fall back to the gap between consecutive vertebrae.
            ordered = sorted(bands.items(), key=lambda kv: kv[1][0], reverse=True)
            for (upper_label, upper_band), (lower_label, lower_band) in pairwise(ordered):
                gap_lo, gap_hi = lower_band[1] + 1, upper_band[0] - 1
                if gap_hi < gap_lo:
                    continue  # vertebrae touch or overlap; no resolvable disc band
                name = f"{vertebra_name(upper_label)}-{vertebra_name(lower_label)}"
                row = measure_band(upper_label, name, gap_lo, gap_hi, is_disc=True)
                if row is not None:
                    result.levels.append(row)

    if not result.levels:
        result.warnings.append("no level had enough canal voxels to measure")
        log.print("No canal levels could be measured", Log_Type.WARNING)

    return result
