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
        warnings (list[str]): Reasons to distrust the numbers.
    """

    group_volumes_mm3: dict[str, float] = field(default_factory=dict)
    per_label_volumes_mm3: dict[str, float] = field(default_factory=dict)
    laterality: dict[str, float] = field(default_factory=dict)
    voxel_size_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    truncated_labels: list[str] = field(default_factory=list)
    lr_coverage_mm: float = 0.0
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


def measure_soft_tissue(vibe_seg: Union[NII, str, Path]) -> SoftTissueVolumes:
    """Reads paraspinal muscle and fat volumes out of a VIBE whole-body segmentation.

    Args:
        vibe_seg (NII | str | Path): The VIBE segmentation the pipeline wrote while cropping, or an
            already-loaded NII of it.

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

    present = {int(v) for v in np.unique(arr) if v != 0}

    # Reject what can be proven not to be VIBE, because the alternative is reporting the spinal cord as
    # paraspinal muscle (see the module docstring: values 59-62 mean different things in each vocabulary).
    max_vibe_value = max(name_to_value.values())
    out_of_range = {v for v in present if v > max_vibe_value}
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

    # Which labels run into the left or right edge of the imaged volume, and so are only partly captured.
    for label_name, value in name_to_value.items():
        mask = arr == value
        if not mask.any():
            continue
        occupied = np.flatnonzero(mask.any(axis=(1, 2)))
        if occupied[0] == 0 or occupied[-1] == arr.shape[0] - 1:
            result.truncated_labels.append(label_name)

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

        # Left/right ratio, but only when neither side is cut off by the edge of the imaged volume. On a
        # sagittal stack the muscles run out of the slab, so the ratio would report slab centring rather
        # than anatomy: a real lumbar study gave 0.514 purely because the left muscle got 6 sagittal slices
        # and the right got 9.
        left = sum(v for k, v in result.per_label_volumes_mm3.items() if k in label_names and k.endswith("_left"))
        right = sum(v for k, v in result.per_label_volumes_mm3.items() if k in label_names and k.endswith("_right"))
        pair_truncated = any(name in result.truncated_labels for name in label_names)
        if left and right and not pair_truncated:
            result.laterality[group] = left / right

    result.warnings.append(
        "the VIBE model is trained for VIBE/Dixon acquisitions and is run here only to place a crop box; "
        "these volumes are unvalidated on sagittal T2w and must be checked before being used as measurements"
    )
    truncated_soft_tissue = [name for names in SOFT_TISSUE_GROUPS.values() for name in names if name in result.truncated_labels]
    if truncated_soft_tissue:
        result.warnings.append(
            f"left-right coverage is only {result.lr_coverage_mm:.0f} mm and these labels reach a slab edge: "
            f"{', '.join(sorted(truncated_soft_tissue))}. The volumes are therefore the part of each "
            "structure inside the imaged slab, not the whole structure, so they are not comparable between "
            "subjects unless the slab is identical. Laterality ratios are omitted for truncated pairs "
            "because they would measure slab centring rather than the patient."
        )
    if spacing[0] > 2.0 or spacing[2] > 2.0:
        result.warnings.append(
            f"voxel spacing {spacing} is coarse for muscle quantification; the paraspinal compartment "
            "extends in the direction that is worst resolved here"
        )
    return result
