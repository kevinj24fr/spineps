"""Flags saying when a SPINEPS level numbering should not be trusted.

Why this exists
---------------
SPINEPS always emits a confident vertebra name, including when the anatomy is genuinely ambiguous. In a
small clinical cohort it called a transitional level -- a thirteenth thoracic or a sixth lumbar vertebra --
in four of five patients. Whether that call is right, and *where* the extra level was placed, changes
everything downstream:

* An extra level assigned at the **top** of the lumbar spine (T13) leaves lumbar numbering intact.
* An extra level assigned at the **bottom** (L6) shifts every per-level result by one, so an "L4-L5"
  measurement is really L3-L4.

Silently propagating that over a large dataset produces results that are wrong in a way nobody notices.
This module reports the conditions that make a numbering unreliable so downstream code can withhold trust
rather than guess.

What it can and cannot tell you
-------------------------------
These are *observable* conditions read off the finished masks: whether the sacrum was visible to anchor the
count from below, whether the field of view is truncated at either end, and whether a transitional level
was called and where. They do not re-run the labelling solver, so they are not a posterior probability --
a numbering with no flags raised can still be wrong. Absence of flags is weaker evidence than presence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from TPTBox import NII
from TPTBox.core.vert_constants import v_name2idx

from spineps.metrics.labels import decode_vertebra_label, vertebra_name

# Instance labels of the transitional vertebrae SPINEPS can assign. Their presence is the signal that the
# count depended on a judgement call rather than on unambiguous anatomy.
TRANSITIONAL_LABELS: dict[int, str] = {
    v_name2idx["T13"]: "T13",
    v_name2idx["L6"]: "L6",
}

#: Sacrum instance label, used to check whether the count is anchored from below.
SACRUM_LABEL = v_name2idx["S1"]

#: A mask touching within this many voxels of a volume edge counts as truncated there.
EDGE_TOLERANCE_VOXELS = 1


@dataclass
class NumberingConfidence:
    """Conditions bearing on whether a level numbering can be trusted.

    Attributes:
        trustworthy (bool): False if any condition was found that can shift the numbering.
        reasons (list[str]): Human-readable description of each condition found.
        transitional_called (str | None): Name of the transitional vertebra assigned, if any.
        transitional_shifts_lumbar (bool): True if the transitional level sits below the lumbar spine and
            therefore renumbers every lumbar level.
        sacrum_visible (bool): Whether a sacrum was segmented to anchor the count from below.
        truncated_superior (bool): Whether the segmentation runs into the top edge of the volume.
        truncated_inferior (bool): Whether it runs into the bottom edge.
        n_levels (int): Number of distinct vertebra instances found.
        levels (list[str]): The assigned level names, cranio-caudal.
    """

    trustworthy: bool = True
    reasons: list[str] = field(default_factory=list)
    transitional_called: str | None = None
    transitional_shifts_lumbar: bool = False
    sacrum_visible: bool = False
    truncated_superior: bool = False
    truncated_inferior: bool = False
    n_levels: int = 0
    levels: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        """Flattens the assessment into a dict suitable for a TSV row or JSON block.

        Returns:
            dict: One key per field, with ``reasons`` and ``levels`` joined into strings.
        """
        return {
            "numbering_trustworthy": self.trustworthy,
            "numbering_reasons": "; ".join(self.reasons) if self.reasons else "",
            "transitional_called": self.transitional_called or "",
            "transitional_shifts_lumbar": self.transitional_shifts_lumbar,
            "sacrum_visible": self.sacrum_visible,
            "truncated_superior": self.truncated_superior,
            "truncated_inferior": self.truncated_inferior,
            "n_levels": self.n_levels,
            "levels": ",".join(self.levels),
        }


def assess_numbering(vert_nii: NII) -> NumberingConfidence:
    """Reports the conditions that make a SPINEPS level numbering unreliable.

    Args:
        vert_nii (NII): Vertebra instance mask, as written by the pipeline.

    Returns:
        NumberingConfidence: The flags, with ``trustworthy`` False if any were raised.
    """
    # Canonical orientation so "superior" is unambiguously the far end of the last axis.
    verts = vert_nii.copy().reorient_(("R", "A", "S"))
    arr = verts.get_seg_array().astype(np.int32)

    raw_labels = [int(v) for v in np.unique(arr) if v != 0]
    # The mask stores a vertebra's disc and endplate as the same index offset by multiples of 100, so
    # collapse to vertebra identity before counting levels or looking for transitional vertebrae.
    vertebra_labels = sorted({decode_vertebra_label(v)[0] for v in raw_labels})

    # Order cranio-caudally by the top of each vertebra's extent, matching on the decoded index so a
    # vertebra's disc and endplate voxels count towards its position.
    decoded = np.zeros_like(arr)
    for raw in raw_labels:
        decoded[arr == raw] = decode_vertebra_label(raw)[0]

    def top_of(index: int) -> int:
        occupied = np.flatnonzero((decoded == index).any(axis=(0, 1)))
        return int(occupied[-1]) if occupied.size else -1

    ordered = sorted(vertebra_labels, key=top_of, reverse=True)

    result = NumberingConfidence(
        n_levels=len(vertebra_labels),
        levels=[vertebra_name(v) for v in ordered],
        sacrum_visible=SACRUM_LABEL in vertebra_labels,
    )
    labels = vertebra_labels

    # A transitional level means the count rested on a judgement call.
    for label, name in TRANSITIONAL_LABELS.items():
        if label in labels:
            result.transitional_called = name
            # L6 sits below the lumbar spine, so it renumbers everything above it; T13 does not.
            result.transitional_shifts_lumbar = name == "L6"
            shift_note = (
                "every lumbar level below it is renumbered" if result.transitional_shifts_lumbar
                else "lumbar numbering is unaffected"
            )
            result.reasons.append(f"transitional vertebra {name} was assigned, so {shift_note}")
            result.trustworthy = False

    if not result.sacrum_visible:
        result.reasons.append(
            "no sacrum was segmented, so the count is not anchored from below and a whole-spine "
            "off-by-one cannot be ruled out"
        )
        result.trustworthy = False

    # Truncation: does the segmentation run into either end of the acquired volume?
    occupied_is = np.flatnonzero((arr > 0).any(axis=(0, 1)))
    if occupied_is.size:
        if occupied_is[-1] >= arr.shape[2] - 1 - EDGE_TOLERANCE_VOXELS:
            result.truncated_superior = True
            result.reasons.append("segmentation reaches the superior edge of the volume; levels may be cut off above")
            result.trustworthy = False
        if occupied_is[0] <= EDGE_TOLERANCE_VOXELS:
            result.truncated_inferior = True
            result.reasons.append("segmentation reaches the inferior edge of the volume; levels may be cut off below")
            result.trustworthy = False
    else:
        result.reasons.append("no vertebra instances found")
        result.trustworthy = False

    return result
