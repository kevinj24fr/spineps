"""Paraspinal muscle and fat volumes from the whole-body segmentation SPINEPS already computes.

Where this comes from
---------------------
When SPINEPS crops to the spine it runs the VIBE whole-body model, writes the full segmentation to disk,
and then uses exactly four of its labels (disc, vertebra body, posterior elements, sacrum) to build a
bounding box. The other 68 labels are written and never looked at again -- including
``autochthon_left``/``autochthon_right``, which is the erector spinae group, the paraspinal muscle usually
meant when spinal muscle is discussed as a biomarker.

So these volumes cost no extra inference. They are already in the ``*_seg-vibe_msk.nii.gz`` the pipeline
leaves behind.

Read this before using the numbers
----------------------------------
The VIBE model is trained for VIBE/Dixon acquisitions. SPINEPS runs it on whatever it was given -- for the
T2w pipeline, a sagittal T2w -- because a bounding box only needs to be roughly right. **Nobody has shown
that these labels are accurate enough on sagittal T2w to quantify muscle or fat.** Two specific concerns:

* A sagittal acquisition with 3-4 mm slices resolves the paraspinal compartment poorly in the left-right
  direction, which is exactly the direction its cross-section extends.
* Out-of-distribution input can produce confident, wrong labels rather than no labels.

Validate against a reader or a dedicated axial acquisition on your own data before treating any of this as
a measurement. The function reports what the model claims; it cannot tell you whether the claim is true.

Do not pass the wrong mask
--------------------------
The VIBE label values and SPINEPS' own semantic label values occupy the same integers with entirely
different meanings, and the overlap lands exactly on the labels this module reads::

    value 59  VIBE autochthon_left   == SPINEPS Vertebra_Disc_Inferior
    value 60  VIBE autochthon_right  == SPINEPS Spinal_Cord
    value 61  VIBE iliopsoas_left    == SPINEPS Spinal_Canal
    value 62  VIBE iliopsoas_right   == SPINEPS Endplate

So handing a ``seg-spine_msk`` to this function would report the spinal cord as paraspinal muscle and the
spinal canal as psoas, with entirely plausible numbers. :func:`measure_soft_tissue` rejects inputs it can
prove are not VIBE (any label above the VIBE range, or the SPINEPS disc label), but a mask containing
*only* colliding values is genuinely indistinguishable from content alone. Pass the
``*_seg-vibe_msk.nii.gz`` the pipeline wrote, not a mask you assembled yourself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
from TPTBox import NII, to_nii
from TPTBox.core.vert_constants import Location

from spineps.metrics.labels import decode_vertebra_label, vertebra_name

#: Midline structures used to locate the mid-sagittal plane, in order of preference. These sit at the
#: centre of the body and are narrow, so they are the labels least likely to be cut off by a sagittal slab.
MIDLINE_ANCHORS: tuple[str, ...] = ("spinal_channel", "spinal_cord", "vertebra_body")

#: Default half-width, either side of the midline, that per-level volumes are measured over. Fixing this
#: is what makes two subjects comparable: a volume measured over "whatever the slab happened to cover"
#: tracks the field of view, not the patient. On a real cohort the available half-width ranged from 22 to
#: 56 mm, so 20 mm is supplied by every study; stations that cannot supply it are flagged instead of
#: silently measured over less.
DEFAULT_REFERENCE_HALF_WIDTH_MM = 20.0

#: VIBE labels grouped into the compartments worth reporting for spine work.
SOFT_TISSUE_GROUPS: dict[str, tuple[str, ...]] = {
    "paraspinal_muscle": ("autochthon_left", "autochthon_right"),
    "psoas": ("iliopsoas_left", "iliopsoas_right"),
    "gluteal_muscle": (
        "gluteus_maximus_left",
        "gluteus_maximus_right",
        "gluteus_medius_left",
        "gluteus_medius_right",
        "gluteus_minimus_left",
        "gluteus_minimus_right",
    ),
    "subcutaneous_fat": ("subcutaneous_fat",),
    "inner_fat": ("inner_fat",),
    "other_muscle": ("muscle",),
}


@dataclass
class SoftTissueVolumes:
    """Soft-tissue volumes read out of a VIBE whole-body segmentation.

    Attributes:
        group_volumes_mm3 (dict[str, float]): Volume per compartment in SOFT_TISSUE_GROUPS. On a sagittal
            acquisition these are the volume *inside the imaged slab*, not the whole muscle.
        per_label_volumes_mm3 (dict[str, float]): Volume for each individual VIBE label found.
        laterality (dict[str, float]): For paired compartments, the left/right volume ratio; 1.0 means
            symmetric. Omitted for any pair that is cut off at a slab edge, because the ratio would then
            measure where the slab sits rather than the patient.
        voxel_size_mm (tuple[float, float, float]): Voxel spacing of the segmentation measured.
        truncated_labels (list[str]): Labels reaching the left or right edge of the volume, and therefore
            only partially imaged.
        lr_coverage_mm (float): Total left-right extent of the imaged volume.
        symmetric_half_width_mm (float): Distance either side of the midline over which the left/right
            comparison was made. 0 if no midline anchor was found.
        reference_half_width_mm (float): Fixed half-width the per-level volumes were measured over, or 0 if
            the slab could not supply it (in which case those volumes are not cross-subject comparable).
        midline_lr_index (int): Left-right index taken as the mid-sagittal plane, or -1 if not found.
        per_level_mm3 (dict[str, dict[str, float]]): Compartment volume per vertebral level, present only
            when a vertebra mask was supplied. Keyed level name -> compartment -> volume.
        warnings (list[str]): Reasons to distrust the numbers.
    """

    group_volumes_mm3: dict[str, float] = field(default_factory=dict)
    per_label_volumes_mm3: dict[str, float] = field(default_factory=dict)
    laterality: dict[str, float] = field(default_factory=dict)
    voxel_size_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    truncated_labels: list[str] = field(default_factory=list)
    lr_coverage_mm: float = 0.0
    symmetric_half_width_mm: float = 0.0
    reference_half_width_mm: float = 0.0
    midline_lr_index: int = -1
    per_level_mm3: dict[str, dict[str, float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        """Flattens the compartment volumes into a dict suitable for a TSV row.

        Returns:
            dict: One key per compartment volume, plus laterality ratios.
        """
        row: dict = {f"{name}_mm3": round(v, 3) for name, v in sorted(self.group_volumes_mm3.items())}
        row.update({f"{name}_lr_ratio": round(v, 4) for name, v in sorted(self.laterality.items())})
        return row


def _vibe_label_values() -> dict[str, int]:
    """Builds a name-to-value map for the VIBE whole-body label set.

    Returns:
        dict[str, int]: VIBE label names mapped to their integer values.
    """
    from TPTBox.core.vert_constants import Full_Body_Instance_Vibe

    return {member.name: int(member.value) for member in Full_Body_Instance_Vibe}


def _midline_lr_index(arr: np.ndarray, name_to_value: dict[str, int]) -> Union[int, None]:
    """Locates the mid-sagittal plane from a midline structure.

    Args:
        arr (np.ndarray): VIBE label array in (LR, PA, IS) axis order.
        name_to_value (dict[str, int]): VIBE label name to value map.

    Returns:
        int | None: Left-right index of the midline, or None if no anchor was found.
    """
    for anchor in MIDLINE_ANCHORS:
        value = name_to_value.get(anchor)
        if value is None:
            continue
        occupied = np.flatnonzero((arr == value).any(axis=(1, 2)))
        if occupied.size:
            return round((int(occupied[0]) + int(occupied[-1])) / 2)
    return None


def _reject_non_vibe(arr: np.ndarray, name_to_value: dict[str, int]) -> None:
    """Raises if the array can be shown not to be a VIBE whole-body segmentation.

    The check matters because VIBE and SPINEPS label values collide on exactly the labels this module reads,
    so the failure mode is not an error but a plausible wrong answer: SPINEPS' spinal cord reported as
    paraspinal muscle.

    Args:
        arr (np.ndarray): Label array to check.
        name_to_value (dict[str, int]): VIBE label name to value map.

    Raises:
        ValueError: If a label outside the VIBE range is present, or no soft-tissue label is.
    """
    present = {int(v) for v in np.unique(arr) if v != 0}
    out_of_range = {v for v in present if v > max(name_to_value.values())}
    spineps_disc = int(Location.Vertebra_Disc.value)
    if out_of_range or spineps_disc in present:
        offending = sorted(out_of_range) or [spineps_disc]
        raise ValueError(
            f"this is not a VIBE whole-body segmentation: label(s) {offending[:8]} cannot occur in one. "
            "It looks like a SPINEPS semantic mask, whose label values collide with VIBE's on exactly the "
            "soft-tissue labels this reads (60 is Spinal_Cord there and autochthon_right here). Pass the "
            "*_seg-vibe_msk.nii.gz the pipeline wrote while cropping."
        )

    wanted = {name_to_value[n] for names in SOFT_TISSUE_GROUPS.values() for n in names if n in name_to_value}
    if not present.intersection(wanted):
        raise ValueError(
            "segmentation contains none of the VIBE soft-tissue labels; this does not look like a VIBE "
            f"whole-body segmentation (found labels {sorted(present)[:12]}...)"
        )


def _reference_window(
    result: SoftTissueVolumes,
    arr: np.ndarray,
    spacing_lr: float,
    midline: Union[int, None],
    reference_half_width_mm: float,
) -> Union[tuple[int, int], None]:
    """Returns the fixed left-right slice window per-level volumes should be measured over.

    Fixing the window is what makes a level volume comparable between subjects. Measured over "whatever the
    slab covered", the number is partly a measure of how wide the acquisition was: on a real cohort L3
    paraspinal volume tracked left-right coverage almost monotonically.

    Args:
        result (SoftTissueVolumes): Result object; a warning is appended if the window cannot be supplied.
        arr (np.ndarray): VIBE label array in (LR, PA, IS) axis order.
        spacing_lr (float): Voxel spacing along the left-right axis in mm.
        midline (int | None): Left-right index of the mid-sagittal plane.
        reference_half_width_mm (float): Requested half-width either side of the midline.

    Returns:
        tuple[int, int] | None: Inclusive slice range, or None if the slab is too narrow to supply it.
    """
    if midline is None or spacing_lr <= 0:
        return None
    half_slices = round(reference_half_width_mm / spacing_lr)
    lo, hi = midline - half_slices, midline + half_slices
    if lo >= 0 and hi <= arr.shape[0] - 1:
        result.reference_half_width_mm = reference_half_width_mm
        return lo, hi
    available = min(midline, arr.shape[0] - 1 - midline) * spacing_lr
    result.warnings.append(
        f"left-right coverage supplies only +/-{available:.0f} mm about the midline, less than the "
        f"+/-{reference_half_width_mm:.0f} mm reference window, so per-level volumes are measured over the "
        "whole slab and are NOT comparable with other subjects"
    )
    return None


def _add_per_level_volumes(
    result: SoftTissueVolumes,
    arr: np.ndarray,
    vert_nii: NII,
    name_to_value: dict[str, int],
    voxel_volume: float,
    lr_window: Union[tuple[int, int], None],
) -> None:
    """Adds compartment volumes per vertebral level to ``result``.

    Volumes normalised to a vertebral level are anchored to anatomy rather than to the field of view, so two
    subjects whose slabs cover different amounts of spine remain comparable level for level. Raw whole-slab
    volumes do not have that property.

    Args:
        result (SoftTissueVolumes): Result object to populate in place.
        arr (np.ndarray): VIBE label array in (LR, PA, IS) axis order.
        vert_nii (NII): Vertebra instance mask.
        name_to_value (dict[str, int]): VIBE label name to value map.
        voxel_volume (float): Volume of one voxel in mm^3.
        lr_window (tuple[int, int] | None): Inclusive left-right slice range to restrict to, so the volume
            covers the same geometry in every subject. None measures over the whole slab, which is not
            comparable between subjects.
    """
    verts = vert_nii.copy().reorient_(("R", "A", "S")).get_seg_array().astype(np.int32)
    if verts.shape != arr.shape:
        result.warnings.append(
            f"vertebra mask shape {verts.shape} does not match the VIBE mask {arr.shape}; per-level volumes were skipped"
        )
        return

    # Keep only the vertebra bodies: the mask also stores each level's disc and endplate offset by
    # multiples of 100, and those bands overlap.
    vertebra_only = np.zeros_like(verts)
    for raw in (int(v) for v in np.unique(verts) if v != 0):
        index, subregion = decode_vertebra_label(raw)
        if subregion == 0:
            vertebra_only[verts == raw] = index

    for index in (int(v) for v in np.unique(vertebra_only) if v != 0):
        occupied = np.flatnonzero((vertebra_only == index).any(axis=(0, 1)))
        if not occupied.size:
            continue
        band = arr[:, :, int(occupied[0]) : int(occupied[-1]) + 1]
        if lr_window is not None:
            band = band[lr_window[0] : lr_window[1] + 1]
        per_group = {}
        for group, label_names in SOFT_TISSUE_GROUPS.items():
            voxels = 0
            for label_name in label_names:
                value = name_to_value.get(label_name)
                if value is not None:
                    voxels += int((band == value).sum())
            if voxels:
                per_group[group] = voxels * voxel_volume
        if per_group:
            result.per_level_mm3[vertebra_name(index)] = per_group


def measure_soft_tissue(
    vibe_seg: Union[NII, str, Path],
    vert_nii: Union[NII, None] = None,
    reference_half_width_mm: float = DEFAULT_REFERENCE_HALF_WIDTH_MM,
) -> SoftTissueVolumes:
    """Reads paraspinal muscle and fat volumes out of a VIBE whole-body segmentation.

    Args:
        vibe_seg (NII | str | Path): The VIBE segmentation the pipeline wrote while cropping, or an
            already-loaded NII of it.
        vert_nii (NII | None, optional): Vertebra instance mask. When given, volumes are additionally
            reported per vertebral level, which makes them comparable between subjects whose slabs cover
            different amounts of spine. Defaults to None.
        reference_half_width_mm (float, optional): Distance either side of midline that per-level volumes
            are measured over, so the geometry is identical across subjects. Defaults to
            DEFAULT_REFERENCE_HALF_WIDTH_MM.

    Returns:
        SoftTissueVolumes: Per-compartment volumes, with warnings about their reliability.

    Raises:
        FileNotFoundError: If a path was given and it does not exist.
        ValueError: If the segmentation contains none of the expected soft-tissue labels, which usually
            means it is not a VIBE whole-body segmentation.
    """
    if isinstance(vibe_seg, (str, Path)):
        path = Path(vibe_seg)
        if not path.is_file():
            raise FileNotFoundError(
                f"no VIBE segmentation at {path}. It is only written when the pipeline crops to the spine, "
                "which does not happen for every input."
            )
        seg = to_nii(path, True)
    else:
        seg = vibe_seg

    name_to_value = _vibe_label_values()
    # Reorient so axis 0 is unambiguously left-right; truncation of a paired muscle happens along it.
    seg = seg.copy().reorient_(("R", "A", "S"))
    arr = seg.get_seg_array().astype(np.int32)
    spacing = tuple(float(z) for z in seg.zoom)
    voxel_volume = spacing[0] * spacing[1] * spacing[2]

    result = SoftTissueVolumes(  # type: ignore[arg-type]
        voxel_size_mm=spacing,
        lr_coverage_mm=arr.shape[0] * spacing[0],
    )

    _reject_non_vibe(arr, name_to_value)

    # Which labels run into the left or right edge of the imaged volume, and so are only partly captured.
    for label_name, value in name_to_value.items():
        mask = arr == value
        if not mask.any():
            continue
        occupied = np.flatnonzero(mask.any(axis=(1, 2)))
        if occupied[0] == 0 or occupied[-1] == arr.shape[0] - 1:
            result.truncated_labels.append(label_name)

    # Largest window centred on the midline that fits inside the imaged volume. Both halves then cover the
    # same distance from midline, so a left/right ratio over it is a property of the patient.
    midline = _midline_lr_index(arr, name_to_value)
    window: Union[tuple[int, int, int], None] = None
    if midline is not None:
        half = min(midline, arr.shape[0] - 1 - midline)
        if half > 0:
            window = (midline - half, midline + half, midline)
            result.symmetric_half_width_mm = half * spacing[0]
            result.midline_lr_index = midline

    for group, label_names in SOFT_TISSUE_GROUPS.items():
        total = 0.0
        for label_name in label_names:
            value = name_to_value.get(label_name)
            if value is None:
                continue
            voxels = int((arr == value).sum())
            if voxels:
                volume = voxels * voxel_volume
                result.per_label_volumes_mm3[label_name] = volume
                total += volume
        if total:
            result.group_volumes_mm3[group] = total

        # Left/right ratio measured over mirror-symmetric windows about the midline, rather than over the
        # whole structures. Comparing whole structures on a sagittal slab reports where the slab sits: a
        # real lumbar study gave 0.514 purely because the left muscle got 6 sagittal slices and the right
        # got 9. Restricting both sides to the same distance from midline removes that, and works on a
        # truncated slab instead of requiring an untruncated one.
        if window is not None:
            lo, hi, mid = window
            left_voxels = 0
            right_voxels = 0
            for label_name in label_names:
                value = name_to_value.get(label_name)
                if value is None:
                    continue
                # Exclude the midline slice itself so both halves cover exactly `half` slices.
                left_voxels += int((arr[lo:mid] == value).sum())
                right_voxels += int((arr[mid + 1 : hi + 1] == value).sum())
            if left_voxels and right_voxels:
                result.laterality[group] = left_voxels / right_voxels

    if vert_nii is not None:
        reference_window = _reference_window(result, arr, spacing[0], midline, reference_half_width_mm)
        _add_per_level_volumes(result, arr, vert_nii, name_to_value, voxel_volume, reference_window)

    result.warnings.append(
        "the VIBE model is trained for VIBE/Dixon acquisitions and is run here only to place a crop box; "
        "these volumes are unvalidated on sagittal T2w and must be checked before being used as measurements"
    )
    truncated_soft_tissue = [name for names in SOFT_TISSUE_GROUPS.values() for name in names if name in result.truncated_labels]
    if truncated_soft_tissue:
        result.warnings.append(
            f"left-right coverage is {result.lr_coverage_mm:.0f} mm and these labels reach a slab edge: "
            f"{', '.join(sorted(truncated_soft_tissue))}. The raw volumes are therefore the part of each "
            "structure inside the slab, so do NOT compare them between subjects whose coverage differs. "
            "Use the laterality ratios, which are measured over mirror-symmetric windows about the midline, "
            "and per_level_mm3, which is normalised to anatomy rather than to the field of view."
        )
    if spacing[0] > 2.0 or spacing[2] > 2.0:
        result.warnings.append(
            f"voxel spacing {spacing} is coarse for muscle quantification; the paraspinal compartment "
            "extends in the direction that is worst resolved here"
        )
    return result
