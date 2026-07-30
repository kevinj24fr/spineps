"""Comparison of a canal measurement against a per-level reference distribution.

The point of this module is to stop absolute thresholds being applied across levels. Canal AP diameter is
not constant along the spine -- it rises from roughly 10 mm at the most caudal lumbar level to roughly
14 mm around L1 -- so "below 8 mm" means something different at every level. Comparing a measurement to its
own level's distribution says how unusual it is; comparing it to a fixed number mostly reports which level
it came from.

The bundled distribution is described in ``reference_data/README.md``, including its caveats: it is derived
from a low-back-pain cohort on sagittal acquisitions, and no clinical threshold is implied.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Optional

REFERENCE_FILE = Path(__file__).with_name("reference_data") / "spider_normative_canal.tsv"

#: Offset to subtract from a SPINEPS-derived diameter before comparing it to the bundled distribution.
#:
#: The bundled reference was measured on *human* canal annotations. SPINEPS' canal label is systematically
#: wider: over 278 paired levels from 40 studies, measured with identical level bands so only the canal
#: source differed, SPINEPS read +1.20 mm larger on average (mean absolute difference 1.32 mm, r = +0.913).
#: The offset is near-constant across levels (+0.64 to +1.59 mm) with no caudal-cranial drift, which is what
#: makes a single correction defensible.
#:
#: Comparing an uncorrected SPINEPS measurement against the human-derived distribution would make every
#: level look roughly a millimetre roomier than it is -- a systematic bias towards missing narrowing.
#:
#: This offset remains the weakest number in the package. SPINEPS was trained on 179 of SPIDER's 218
#: subjects and the estimate did not exclude them, so it is optimistic. It cannot be re-estimated on the
#: RSNA lumbar dataset either: RSNA supplies stenosis *grades* and point coordinates, not segmentation
#: masks, so there is nothing there to compare a canal outline against. Re-estimating it needs another
#: dataset with human canal segmentations that SPINEPS has not seen.
SPINEPS_CANAL_OFFSET_MM = 1.20

#: Percentile columns available in the reference table, in order.
PERCENTILES = (5, 25, 50, 75, 95)


@dataclass(frozen=True)
class LevelReference:
    """Reference percentiles of canal AP diameter for one level.

    Attributes:
        index (int): Level index, 1 = most caudal, matching the reference table's numbering.
        structure (str): "vertebra" or "disc".
        n (int): Studies the row was computed from.
        values (dict[int, float]): Percentile to diameter in mm.
    """

    index: int
    structure: str
    n: int
    values: dict[int, float]


@lru_cache(maxsize=1)
def load_reference() -> dict[tuple[int, str], LevelReference]:
    """Loads the bundled per-level reference distribution.

    Returns:
        dict[tuple[int, str], LevelReference]: Keyed by (level index, structure).

    Raises:
        FileNotFoundError: If the bundled reference table is missing from the installation.
    """
    if not REFERENCE_FILE.is_file():
        raise FileNotFoundError(f"bundled reference distribution not found at {REFERENCE_FILE}")
    table: dict[tuple[int, str], LevelReference] = {}
    with REFERENCE_FILE.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            index, structure = int(row["spider_vertebra_index"]), row["structure"]
            table[(index, structure)] = LevelReference(
                index=index,
                structure=structure,
                n=int(row["n"]),
                values={p: float(row[f"p{p}"]) for p in PERCENTILES},
            )
    return table


def calibrate(diameter_mm: float, source: str) -> float:
    """Puts a measurement on the same scale as the bundled reference distribution.

    Args:
        diameter_mm (float): Measured AP diameter.
        source (str): "spineps" for a diameter measured on SPINEPS output, or "reference" for one measured
            on a human annotation (no correction).

    Returns:
        float: The diameter on the reference scale.

    Raises:
        ValueError: If ``source`` is not one of the supported values.
    """
    if source == "reference":
        return diameter_mm
    if source == "spineps":
        return diameter_mm - SPINEPS_CANAL_OFFSET_MM
    raise ValueError(f"source must be 'spineps' or 'reference', got {source!r}")


def percentile_rank(
    diameter_mm: float,
    level_index: int,
    structure: str = "vertebra",
    source: str = "spineps",
) -> Optional[str]:
    """Places a measured diameter within its level's reference distribution.

    Args:
        diameter_mm (float): Measured AP diameter.
        level_index (int): Level index, 1 = most caudal.
        structure (str, optional): "vertebra" or "disc". Defaults to "vertebra".
        source (str, optional): Where the measurement came from, so it can be put on the reference scale.
            Defaults to "spineps", since that is what this package produces.

    Returns:
        str | None: A coarse band such as "<p5", "p25-p50" or ">p95", or None if the reference has no row
            for that level. Deliberately coarse: the table has ~200 studies per level, which does not
            support finer resolution than this.
    """
    row = load_reference().get((int(level_index), structure))
    if row is None:
        return None
    diameter_mm = calibrate(diameter_mm, source)
    ordered = sorted(row.values.items())
    if diameter_mm < ordered[0][1]:
        return f"<p{ordered[0][0]}"
    for (lo_p, lo_v), (hi_p, hi_v) in pairwise(ordered):
        if lo_v <= diameter_mm < hi_v:
            return f"p{lo_p}-p{hi_p}"
    return f">p{ordered[-1][0]}"


def is_unusually_narrow(
    diameter_mm: float,
    level_index: int,
    structure: str = "vertebra",
    source: str = "spineps",
) -> Optional[bool]:
    """Whether a diameter falls below the 5th percentile for its level.

    This is a statement about how uncommon the measurement is in the reference cohort, not a diagnosis. The
    reference cohort is itself made of patients being imaged for low back pain.

    Args:
        diameter_mm (float): Measured AP diameter.
        level_index (int): Level index, 1 = most caudal.
        structure (str, optional): "vertebra" or "disc". Defaults to "vertebra".
        source (str, optional): Where the measurement came from. Defaults to "spineps".

    Returns:
        bool | None: True if below p5, or None if the level is not in the reference.
    """
    row = load_reference().get((int(level_index), structure))
    return None if row is None else calibrate(diameter_mm, source) < row.values[5]
