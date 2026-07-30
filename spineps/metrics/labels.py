"""Decoding of the composite labels SPINEPS writes into its vertebra instance mask.

The ``seg-vert`` mask does not hold bare vertebra indices. Each vertebra contributes up to three values:
the vertebra itself, and the associated disc and endplate offset by multiples of
:data:`SUBREGION_STRIDE`. A real lumbar mask looks like::

    [19, 20, ..., 26,  119, 120, ..., 126,  219, 220, ..., 226]
     \\_ vertebrae _/   \\_ + 100 _______/   \\_ + 200 _______/

So ``label % SUBREGION_STRIDE`` recovers the vertebra identity and ``label // SUBREGION_STRIDE`` says which
part of that level the voxel belongs to. Reading the raw value through the vertebra-name table instead
produces confident nonsense: 113 resolves to "Additional_Vertebral_Body_Middle_Superior_Left" rather than
to T6's disc.
"""

from __future__ import annotations

from TPTBox.core.vert_constants import v_idx2name

#: Multiples of this separate a vertebra's own label from its disc and endplate labels.
SUBREGION_STRIDE = 100

#: Human-readable name per subregion index, for labelling output rows.
SUBREGION_NAMES = {0: "vertebra", 1: "disc", 2: "endplate"}


def decode_vertebra_label(label: int) -> tuple[int, int]:
    """Splits a composite instance label into its vertebra index and subregion index.

    Args:
        label (int): Raw value from the vertebra instance mask.

    Returns:
        tuple[int, int]: ``(vertebra_index, subregion_index)``, where subregion 0 is the vertebra itself.
    """
    value = int(label)
    return value % SUBREGION_STRIDE, value // SUBREGION_STRIDE


def vertebra_name(label: int) -> str:
    """Maps a composite instance label to the name of the vertebra it belongs to.

    Args:
        label (int): Raw value from the vertebra instance mask.

    Returns:
        str: Name such as "L4", or the decoded index as a string if it is not a known vertebra.
    """
    index, _ = decode_vertebra_label(label)
    try:
        return v_idx2name[index]
    except Exception:
        return str(index)


def vertebra_indices(labels: list[int] | set[int]) -> list[int]:
    """Reduces a set of composite labels to the distinct vertebra indices present.

    Args:
        labels (list[int] | set[int]): Raw values from the vertebra instance mask, excluding background.

    Returns:
        list[int]: Sorted distinct vertebra indices.
    """
    return sorted({decode_vertebra_label(v)[0] for v in labels if int(v) != 0})
