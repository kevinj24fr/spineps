# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
"""Tests for the derived measurements.

These build phantoms whose geometry is known by construction, so the assertions check the measured numbers
against arithmetic rather than against whatever the code happens to return.
"""

from __future__ import annotations

import unittest

import nibabel as nib
import numpy as np
from TPTBox import NII
from TPTBox.core.vert_constants import Full_Body_Instance_Vibe, Location, v_name2idx

from spineps.metrics import assess_numbering, measure_canal, measure_soft_tissue
from spineps.metrics.canal import DEFAULT_MIN_AP_EXTENT_MM, MIN_CANAL_VOXELS_PER_LEVEL

# Phantoms are built directly in RAS so axis 0 = L-R, axis 1 = P-A, axis 2 = I-S, matching the
# orientation the measurement code reorients to. Anisotropic on purpose, to catch axis/spacing mix-ups.
ZOOM = (1.0, 0.5, 2.0)
SHAPE = (10, 40, 60)


def _nii(arr: np.ndarray, zoom=ZOOM) -> NII:
    """Wraps an array as a segmentation NII with a diagonal RAS affine."""
    return NII(nib.Nifti1Image(arr.astype(np.uint16), np.diag([*zoom, 1.0])), seg=True)


def _canal_phantom(ap_voxels: int = 8, canal_lr=(4, 6)) -> tuple[NII, NII, list[str]]:
    """Builds a semantic/instance pair with a straight canal of known AP width and three vertebrae.

    Returns:
        tuple[NII, NII, list[str]]: semantic mask, vertebra instance mask, and the level names used.
    """
    semantic = np.zeros(SHAPE, dtype=np.uint16)
    verts = np.zeros(SHAPE, dtype=np.uint16)

    # Canal: constant AP width, spanning the full cranio-caudal extent.
    semantic[canal_lr[0] : canal_lr[1], 20 : 20 + ap_voxels, :] = Location.Spinal_Canal.value

    # Three vertebrae stacked cranio-caudally with 4-voxel gaps between them (the disc levels).
    names = ["L3", "L4", "L5"]
    bands = [(40, 51), (24, 35), (8, 19)]
    for name, (lo, hi) in zip(names, bands):
        verts[2:8, 5:18, lo : hi + 1] = v_name2idx[name]
    return _nii(semantic), _nii(verts), names


class Test_Canal_Geometry(unittest.TestCase):
    def test_ap_diameter_matches_construction(self):
        """A canal 8 voxels wide at 0.5 mm spacing must measure 4.0 mm, not 8 and not 2."""
        semantic, verts, _ = _canal_phantom(ap_voxels=8)
        result = measure_canal(semantic, verts, min_ap_extent_mm=0.0)
        self.assertTrue(result.levels)
        for level in result.levels:
            self.assertAlmostEqual(level.min_ap_diameter_mm, 8 * ZOOM[1], places=6)
            self.assertAlmostEqual(level.mean_ap_diameter_mm, 8 * ZOOM[1], places=6)

    def test_ap_diameter_scales_with_width(self):
        for width in (4, 8, 12):
            semantic, verts, _ = _canal_phantom(ap_voxels=width)
            result = measure_canal(semantic, verts, min_ap_extent_mm=0.0)
            measured = {round(lvl.min_ap_diameter_mm, 6) for lvl in result.levels}
            self.assertEqual(measured, {round(width * ZOOM[1], 6)}, f"width={width}")

    def test_narrowing_is_found_and_reported_as_the_minimum(self):
        """A focal narrowing must show up as the min while leaving the mean higher."""
        semantic, verts, _ = _canal_phantom(ap_voxels=8)
        arr = semantic.get_seg_array()
        # Halve the canal width across two slices inside the L4 band (24..35).
        arr[:, 24:28, 28:30] = 0
        semantic = _nii(arr)
        result = measure_canal(semantic, verts, min_ap_extent_mm=0.0)
        l4 = next(lvl for lvl in result.levels if lvl.level_name == "L4" and not lvl.is_disc_level)
        self.assertLess(l4.min_ap_diameter_mm, l4.mean_ap_diameter_mm)
        self.assertAlmostEqual(l4.min_ap_diameter_mm, 4 * ZOOM[1], places=6)

    def test_interior_hole_does_not_narrow_the_diameter(self):
        """AP diameter is a span, so a segmentation hole must not read as a narrower canal."""
        semantic, verts, _ = _canal_phantom(ap_voxels=8)
        arr = semantic.get_seg_array()
        arr[:, 23:25, :] = 0  # punch a hole in the middle of the canal
        result = measure_canal(_nii(arr), verts, min_ap_extent_mm=0.0)
        for level in result.levels:
            self.assertAlmostEqual(level.min_ap_diameter_mm, 8 * ZOOM[1], places=6)

    def test_volume_matches_voxel_count(self):
        semantic, verts, _ = _canal_phantom(ap_voxels=8)
        result = measure_canal(semantic, verts, include_disc_levels=False)
        voxel_volume = ZOOM[0] * ZOOM[1] * ZOOM[2]
        for level in result.levels:
            self.assertAlmostEqual(level.canal_volume_mm3, level.canal_voxels * voxel_volume, places=6)

    def test_disc_levels_are_reported_and_named_between_vertebrae(self):
        semantic, verts, _ = _canal_phantom()
        result = measure_canal(semantic, verts, include_disc_levels=True)
        disc_names = {lvl.level_name for lvl in result.levels if lvl.is_disc_level}
        self.assertIn("L3-L4", disc_names)
        self.assertIn("L4-L5", disc_names)

    def test_disc_levels_can_be_disabled(self):
        semantic, verts, _ = _canal_phantom()
        result = measure_canal(semantic, verts, include_disc_levels=False)
        self.assertFalse([lvl for lvl in result.levels if lvl.is_disc_level])

    def test_thick_slices_raise_a_warning(self):
        """A thick sagittal acquisition cannot resolve off-midline narrowing; that must be stated."""
        semantic, verts, _ = _canal_phantom()
        thick = (3.3, 0.5, 2.0)
        result = measure_canal(_nii(semantic.get_seg_array(), thick), _nii(verts.get_seg_array(), thick))
        self.assertTrue(any("slice spacing" in w for w in result.warnings))

    def test_missing_canal_label_raises_rather_than_returning_zeros(self):
        _, verts, _ = _canal_phantom()
        empty = _nii(np.zeros(SHAPE, dtype=np.uint16))
        with self.assertRaises(ValueError) as ctx:
            measure_canal(empty, verts)
        self.assertIn("canal", str(ctx.exception).lower())

    def test_mismatched_shapes_raise(self):
        semantic, _, _ = _canal_phantom()
        other = _nii(np.zeros((8, 8, 8), dtype=np.uint16))
        with self.assertRaises(ValueError):
            measure_canal(semantic, other)

    def test_levels_below_the_voxel_floor_are_skipped(self):
        semantic, verts, _ = _canal_phantom(ap_voxels=8)
        arr = semantic.get_seg_array()
        arr[:, :, 24:36] = 0  # remove all canal from the L4 band
        result = measure_canal(_nii(arr), verts, include_disc_levels=False)
        self.assertNotIn("L4", {lvl.level_name for lvl in result.levels})
        self.assertGreaterEqual(MIN_CANAL_VOXELS_PER_LEVEL, 1)

    def test_rows_are_sorted_by_median_not_minimum(self):
        """Ordering must use the stable statistic: min differed by up to 13.6 mm between backends."""
        semantic, verts, _ = _canal_phantom(ap_voxels=8)
        arr = semantic.get_seg_array()
        arr[:, 24:28, 28:30] = 0
        rows = measure_canal(_nii(arr), verts, min_ap_extent_mm=0.0).as_rows()
        medians = [r["median_ap_diameter_mm"] for r in rows]
        self.assertEqual(medians, sorted(medians))

    def test_orientation_independence(self):
        """The same anatomy stored in a different orientation must measure the same."""
        semantic, verts, _ = _canal_phantom(ap_voxels=8)
        flipped_sem = semantic.copy().reorient_(("P", "I", "R"))
        flipped_vert = verts.copy().reorient_(("P", "I", "R"))
        a = measure_canal(semantic, verts, include_disc_levels=False)
        b = measure_canal(flipped_sem, flipped_vert, include_disc_levels=False)
        self.assertEqual(
            {lvl.level_name: round(lvl.min_ap_diameter_mm, 4) for lvl in a.levels},
            {lvl.level_name: round(lvl.min_ap_diameter_mm, 4) for lvl in b.levels},
        )


class Test_Numbering_Confidence(unittest.TestCase):
    def _verts(self, names: list[str], touch_top: bool = False, touch_bottom: bool = False) -> NII:
        arr = np.zeros(SHAPE, dtype=np.uint16)
        span = 4
        start = 0 if touch_bottom else 3
        for i, name in enumerate(names):
            lo = start + i * (span + 1)
            arr[2:8, 5:18, lo : lo + span] = v_name2idx[name]
        if touch_top:
            arr[2:8, 5:18, SHAPE[2] - 1] = v_name2idx[names[-1]]
        return _nii(arr)

    def test_clean_case_is_trustworthy(self):
        verts = self._verts(["L3", "L4", "L5", "S1"])
        result = assess_numbering(verts)
        self.assertTrue(result.trustworthy, result.reasons)
        self.assertTrue(result.sacrum_visible)
        self.assertIsNone(result.transitional_called)

    def test_missing_sacrum_is_flagged(self):
        result = assess_numbering(self._verts(["L3", "L4", "L5"]))
        self.assertFalse(result.trustworthy)
        self.assertFalse(result.sacrum_visible)
        self.assertTrue(any("sacrum" in r for r in result.reasons))

    def test_t13_is_flagged_but_does_not_shift_lumbar(self):
        result = assess_numbering(self._verts(["T13", "L4", "L5", "S1"]))
        self.assertEqual(result.transitional_called, "T13")
        self.assertFalse(result.transitional_shifts_lumbar)
        self.assertFalse(result.trustworthy)
        self.assertTrue(any("unaffected" in r for r in result.reasons))

    def test_l6_is_flagged_as_shifting_lumbar_numbering(self):
        """This is the consequential case: every lumbar level below is renumbered."""
        result = assess_numbering(self._verts(["L4", "L5", "L6", "S1"]))
        self.assertEqual(result.transitional_called, "L6")
        self.assertTrue(result.transitional_shifts_lumbar)
        self.assertFalse(result.trustworthy)
        self.assertTrue(any("renumbered" in r for r in result.reasons))

    def test_truncation_at_each_end_is_detected(self):
        top = assess_numbering(self._verts(["L4", "L5", "S1"], touch_top=True))
        self.assertTrue(top.truncated_superior)
        bottom = assess_numbering(self._verts(["L4", "L5", "S1"], touch_bottom=True))
        self.assertTrue(bottom.truncated_inferior)

    def test_empty_mask_is_not_trustworthy(self):
        result = assess_numbering(_nii(np.zeros(SHAPE, dtype=np.uint16)))
        self.assertFalse(result.trustworthy)
        self.assertEqual(result.n_levels, 0)

    def test_levels_are_listed_cranio_caudally(self):
        result = assess_numbering(self._verts(["L3", "L4", "L5", "S1"]))
        self.assertEqual(result.levels, ["S1", "L5", "L4", "L3"])

    def test_row_serialization_is_flat_and_stringy(self):
        row = assess_numbering(self._verts(["L4", "L5", "S1"])).as_row()
        self.assertIn("numbering_trustworthy", row)
        self.assertIsInstance(row["levels"], str)
        self.assertIsInstance(row["numbering_reasons"], str)


class Test_Soft_Tissue(unittest.TestCase):
    def _vibe(self, counts: dict[str, int], touch_lr_edge: bool = False) -> NII:
        """Builds a VIBE-like segmentation with an exact voxel count per named label.

        Labels are placed away from the left-right edges by default, because touching an edge means the
        structure is only partly imaged and the code then suppresses laterality on purpose.
        """
        arr = np.zeros(SHAPE, dtype=np.uint16)
        lo, hi = (0, SHAPE[0]) if touch_lr_edge else (2, SHAPE[0] - 2)
        interior = arr[lo:hi]
        flat = interior.reshape(-1)
        cursor = 0
        for name, n in counts.items():
            value = int(Full_Body_Instance_Vibe[name].value)
            flat[cursor : cursor + n] = value
            cursor += n
        arr[lo:hi] = flat.reshape(interior.shape)
        return _nii(arr)

    def test_paraspinal_volume_matches_voxel_count(self):
        voxel_volume = ZOOM[0] * ZOOM[1] * ZOOM[2]
        seg = self._vibe({"autochthon_left": 100, "autochthon_right": 150})
        result = measure_soft_tissue(seg)
        self.assertAlmostEqual(result.group_volumes_mm3["paraspinal_muscle"], 250 * voxel_volume, places=6)

    def test_laterality_ratio(self):
        seg = self._vibe({"autochthon_left": 100, "autochthon_right": 200})
        result = measure_soft_tissue(seg)
        self.assertAlmostEqual(result.laterality["paraspinal_muscle"], 0.5, places=6)

    def test_symmetric_gives_unity(self):
        seg = self._vibe({"autochthon_left": 120, "autochthon_right": 120})
        self.assertAlmostEqual(measure_soft_tissue(seg).laterality["paraspinal_muscle"], 1.0, places=6)

    def test_fat_compartments_are_separated(self):
        seg = self._vibe({"autochthon_left": 10, "subcutaneous_fat": 40, "inner_fat": 20})
        result = measure_soft_tissue(seg)
        self.assertIn("subcutaneous_fat", result.group_volumes_mm3)
        self.assertIn("inner_fat", result.group_volumes_mm3)
        self.assertNotAlmostEqual(result.group_volumes_mm3["subcutaneous_fat"], result.group_volumes_mm3["inner_fat"])

    def test_validation_warning_is_always_present(self):
        """The numbers are unvalidated on T2w; that caveat must travel with them."""
        seg = self._vibe({"autochthon_left": 10, "autochthon_right": 10})
        result = measure_soft_tissue(seg)
        self.assertTrue(any("unvalidated" in w for w in result.warnings))

    def test_spineps_semantic_mask_is_rejected(self):
        """The dangerous misuse: value 60 is Spinal_Cord in SPINEPS and autochthon_right in VIBE, so a
        semantic mask would silently report the cord as paraspinal muscle."""
        arr = np.zeros(SHAPE, dtype=np.uint16)
        arr[0, 0, 0] = Location.Spinal_Cord.value  # 60, collides with autochthon_right
        arr[0, 0, 1] = Location.Spinal_Canal.value  # 61, collides with iliopsoas_left
        arr[0, 0, 2] = Location.Vertebra_Disc.value  # 100, impossible in VIBE -> the tell
        with self.assertRaises(ValueError) as ctx:
            measure_soft_tissue(_nii(arr))
        message = str(ctx.exception).lower()
        self.assertIn("not a vibe", message)
        self.assertIn("seg-vibe", message)

    def test_labels_above_the_vibe_range_are_rejected(self):
        arr = np.zeros(SHAPE, dtype=np.uint16)
        arr[0, 0, 0] = int(Full_Body_Instance_Vibe.autochthon_left.value)
        arr[0, 0, 1] = 200  # no VIBE label goes this high
        with self.assertRaises(ValueError):
            measure_soft_tissue(_nii(arr))

    def test_missing_file_raises_with_an_explanation(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            measure_soft_tissue("/nonexistent/vibe_msk.nii.gz")
        self.assertIn("crop", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()


class Test_Canal_AP_Extent_Floor(unittest.TestCase):
    """The taper floor exists because real canals end in a few stray voxels that otherwise set the minimum."""

    def test_floor_excludes_thin_taper_rows(self):
        semantic, verts, _ = _canal_phantom(ap_voxels=8)
        arr = semantic.get_seg_array()
        # A 1-voxel-wide row inside the L4 band, standing in for canal taper: 0.5 mm of AP extent.
        arr[:, 20:21, 30] = Location.Spinal_Canal.value
        arr[:, 21:28, 30] = 0
        with_floor = measure_canal(_nii(arr), verts, min_ap_extent_mm=DEFAULT_MIN_AP_EXTENT_MM)
        without_floor = measure_canal(_nii(arr), verts, min_ap_extent_mm=0.0)
        thin = min(lvl.min_ap_diameter_mm for lvl in without_floor.levels)
        kept = min(lvl.min_ap_diameter_mm for lvl in with_floor.levels)
        self.assertAlmostEqual(thin, 0.5, places=6)
        self.assertGreaterEqual(kept, DEFAULT_MIN_AP_EXTENT_MM)

    def test_default_floor_is_documented_in_mm(self):
        """A voxel-count floor would silently change meaning with resolution; this one is in mm."""
        self.assertIsInstance(DEFAULT_MIN_AP_EXTENT_MM, float)
        self.assertGreater(DEFAULT_MIN_AP_EXTENT_MM, 0.0)


class Test_Numbering_Flag_Calibration(unittest.TestCase):
    """A flag that fires on nearly every scan carries no information.

    On a real 18-station cohort the segmentation reached a volume edge 16 times, because the spinal column
    normally continues past the field of view. Truncation is therefore recorded but must not by itself
    condemn the numbering; what matters is whether the count is anchored by a visible sacrum.
    """

    def _verts(self, names: list[str], touch_top: bool = False, touch_bottom: bool = False) -> NII:
        arr = np.zeros(SHAPE, dtype=np.uint16)
        span = 4
        start = 0 if touch_bottom else 3
        for i, name in enumerate(names):
            lo = start + i * (span + 1)
            arr[2:8, 5:18, lo : lo + span] = v_name2idx[name]
        if touch_top:
            arr[2:8, 5:18, SHAPE[2] - 1] = v_name2idx[names[-1]]
        return _nii(arr)

    def test_truncation_with_a_visible_sacrum_stays_trustworthy(self):
        result = assess_numbering(self._verts(["L4", "L5", "S1"], touch_top=True, touch_bottom=True))
        self.assertTrue(result.truncated_superior)
        self.assertTrue(result.truncated_inferior)
        self.assertTrue(result.trustworthy, result.reasons)

    def test_truncation_without_a_sacrum_is_not_trustworthy(self):
        result = assess_numbering(self._verts(["L3", "L4", "L5"], touch_top=True))
        self.assertFalse(result.trustworthy)
        self.assertTrue(any("anchor" in r for r in result.reasons))

    def test_truncation_is_still_reported_for_context(self):
        result = assess_numbering(self._verts(["L4", "L5", "S1"], touch_top=True))
        self.assertTrue(result.truncated_superior)
        self.assertFalse(result.truncated_inferior)


class Test_Soft_Tissue_Slab_Truncation(unittest.TestCase):
    """A sagittal stack cuts the paraspinal muscles off, so the volumes are slab fragments.

    On a real lumbar study the left-right coverage was 61.6 mm and both autochthon labels reached an edge;
    the resulting left/right ratio of 0.514 reflected the left muscle getting 6 sagittal slices and the
    right getting 9, not the patient.
    """

    def _vibe(self, counts: dict[str, int], touch_lr_edge: bool) -> NII:
        arr = np.zeros(SHAPE, dtype=np.uint16)
        lo, hi = (0, SHAPE[0]) if touch_lr_edge else (2, SHAPE[0] - 2)
        interior = arr[lo:hi]
        flat = interior.reshape(-1)
        cursor = 0
        for name, n in counts.items():
            flat[cursor : cursor + n] = int(Full_Body_Instance_Vibe[name].value)
            cursor += n
        arr[lo:hi] = flat.reshape(interior.shape)
        return _nii(arr)

    def test_edge_touching_labels_are_marked_truncated(self):
        seg = self._vibe({"autochthon_left": 40, "autochthon_right": 40}, touch_lr_edge=True)
        result = measure_soft_tissue(seg)
        self.assertIn("autochthon_left", result.truncated_labels)

    def test_laterality_is_suppressed_when_truncated(self):
        """Reporting a ratio computed from partly-imaged muscle would be worse than reporting nothing."""
        truncated = measure_soft_tissue(self._vibe({"autochthon_left": 40, "autochthon_right": 80}, True))
        intact = measure_soft_tissue(self._vibe({"autochthon_left": 40, "autochthon_right": 80}, False))
        self.assertNotIn("paraspinal_muscle", truncated.laterality)
        self.assertIn("paraspinal_muscle", intact.laterality)
        self.assertAlmostEqual(intact.laterality["paraspinal_muscle"], 0.5, places=6)

    def test_truncation_warning_explains_the_consequence(self):
        result = measure_soft_tissue(self._vibe({"autochthon_left": 40, "autochthon_right": 40}, True))
        joined = " ".join(result.warnings)
        self.assertIn("slab", joined)
        self.assertIn("not comparable between", joined)

    def test_lr_coverage_is_reported(self):
        result = measure_soft_tissue(self._vibe({"autochthon_left": 40, "autochthon_right": 40}, False))
        self.assertAlmostEqual(result.lr_coverage_mm, SHAPE[0] * ZOOM[0], places=6)
