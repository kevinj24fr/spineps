# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import unittest

import torch
from TPTBox.segmentation.nnUnet_utils import predictor as tptbox_predictor

from spineps.utils import tptbox_compat
from spineps.utils.tptbox_compat import _device_kind, patch_nnunet_gpu_memory_helpers


class _Patched:
    """Applies the patch on a pristine copy of TPTBox's helpers and restores them afterwards."""

    def __enter__(self):
        self._orig = (tptbox_predictor.get_gpu_memory_MB, tptbox_predictor.get_gpu_util)
        self._was_patched = tptbox_compat._patched
        tptbox_compat._patched = False
        patch_nnunet_gpu_memory_helpers()
        return self

    def __exit__(self, *exc):
        tptbox_predictor.get_gpu_memory_MB, tptbox_predictor.get_gpu_util = self._orig
        tptbox_compat._patched = self._was_patched
        return False


class Test_DeviceKind(unittest.TestCase):
    def test_normalizes_torch_devices_and_strings(self):
        self.assertEqual(_device_kind(torch.device("mps")), "mps")
        self.assertEqual(_device_kind(torch.device("cuda", 1)), "cuda")
        self.assertEqual(_device_kind("MPS"), "mps")
        self.assertEqual(_device_kind("cuda:2"), "cuda")
        self.assertEqual(_device_kind("cpu"), "cpu")

    def test_none_and_int_are_cuda_like_tptbox_default(self):
        self.assertEqual(_device_kind(None), "cuda")
        self.assertEqual(_device_kind(0), "cuda")


class Test_Patch(unittest.TestCase):
    def test_patch_reports_success_and_is_idempotent(self):
        with _Patched():
            self.assertTrue(patch_nnunet_gpu_memory_helpers())
            self.assertTrue(patch_nnunet_gpu_memory_helpers())

    def test_unpatched_helpers_raise_on_cpu(self):
        """Guards the premise of this module: upstream really does fail on non-CUDA devices."""
        if torch.cuda.is_available():
            self.skipTest("upstream helpers only fail on hosts without CUDA")
        with self.assertRaises(ValueError):
            tptbox_predictor.get_gpu_util(torch.device("cpu"))

    def test_cpu_queries_do_not_raise_after_patching(self):
        with _Patched():
            self.assertEqual(tptbox_predictor.get_gpu_util(torch.device("cpu")), 0.0)
            self.assertGreaterEqual(tptbox_predictor.get_gpu_memory_MB(torch.device("cpu")), 0.0)

    def test_mps_queries_are_in_range(self):
        if not torch.backends.mps.is_available():
            self.skipTest("no Metal device on this host")
        with _Patched():
            util = tptbox_predictor.get_gpu_util(torch.device("mps"))
            memory = tptbox_predictor.get_gpu_memory_MB(torch.device("mps"))
            self.assertGreaterEqual(util, 0.0)
            self.assertLessEqual(util, 1.0)
            self.assertGreater(memory, 0.0)

    def test_cuda_requests_are_delegated_to_the_original(self):
        """CUDA hosts must keep TPTBox's own implementation, not our shim's numbers."""
        calls = []
        orig_mem, orig_util = tptbox_predictor.get_gpu_memory_MB, tptbox_predictor.get_gpu_util
        was_patched = tptbox_compat._patched
        try:
            tptbox_predictor.get_gpu_memory_MB = lambda device: calls.append(("mem", device)) or 1234.0
            tptbox_predictor.get_gpu_util = lambda device: calls.append(("util", device)) or 0.5
            tptbox_compat._patched = False
            patch_nnunet_gpu_memory_helpers()

            self.assertEqual(tptbox_predictor.get_gpu_memory_MB(torch.device("cuda", 0)), 1234.0)
            self.assertEqual(tptbox_predictor.get_gpu_util(torch.device("cuda", 0)), 0.5)
            self.assertEqual([c[0] for c in calls], ["mem", "util"])

            # Non-CUDA devices must not reach the original implementation.
            calls.clear()
            tptbox_predictor.get_gpu_util(torch.device("cpu"))
            self.assertEqual(calls, [])
        finally:
            tptbox_predictor.get_gpu_memory_MB, tptbox_predictor.get_gpu_util = orig_mem, orig_util
            tptbox_compat._patched = was_patched


if __name__ == "__main__":
    unittest.main()
