# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import os
import unittest

import torch

from spineps.utils.device import (
    MPS_FALLBACK_ENV,
    cuda_is_available,
    device_to_ddevice,
    enable_mps_cpu_fallback,
    mps_is_available,
    resolve_device,
)


class _Availability:
    """Context manager faking cuda/mps availability so the tests run on any host."""

    def __init__(self, cuda: bool, mps: bool):
        self.cuda = cuda
        self.mps = mps

    def __enter__(self):
        from spineps.utils import device as device_module

        self._orig = (device_module.cuda_is_available, device_module.mps_is_available)
        device_module.cuda_is_available = lambda: self.cuda
        device_module.mps_is_available = lambda: self.mps
        return self

    def __exit__(self, *exc):
        from spineps.utils import device as device_module

        device_module.cuda_is_available, device_module.mps_is_available = self._orig
        return False


class Test_Availability(unittest.TestCase):
    def test_availability_helpers_return_bool(self):
        self.assertIsInstance(cuda_is_available(), bool)
        self.assertIsInstance(mps_is_available(), bool)

    def test_availability_matches_torch(self):
        self.assertEqual(cuda_is_available(), torch.cuda.is_available())


class Test_ResolveDevice(unittest.TestCase):
    def test_auto_prefers_cuda_over_mps(self):
        with _Availability(cuda=True, mps=True):
            self.assertEqual(resolve_device("auto").type, "cuda")

    def test_auto_picks_mps_when_no_cuda(self):
        with _Availability(cuda=False, mps=True):
            self.assertEqual(resolve_device("auto").type, "mps")

    def test_auto_falls_back_to_cpu(self):
        with _Availability(cuda=False, mps=False):
            self.assertEqual(resolve_device("auto").type, "cpu")

    def test_cuda_gets_explicit_index(self):
        with _Availability(cuda=True, mps=False):
            device = resolve_device("cuda")
            self.assertEqual(device.type, "cuda")
            self.assertEqual(device.index, 0)

    def test_explicit_cpu_is_respected_even_with_accelerators(self):
        with _Availability(cuda=True, mps=True):
            self.assertEqual(resolve_device("cpu").type, "cpu")

    def test_use_cpu_overrides_everything(self):
        with _Availability(cuda=True, mps=True):
            self.assertEqual(resolve_device("auto", use_cpu=True).type, "cpu")
            self.assertEqual(resolve_device("mps", use_cpu=True).type, "cpu")
            self.assertEqual(resolve_device("cuda", use_cpu=True).type, "cpu")

    def test_explicit_cuda_request_raises_when_unavailable(self):
        """Metal must never be silently substituted for CUDA: the two do not agree voxel-for-voxel."""
        for mps_present in (True, False):
            with _Availability(cuda=False, mps=mps_present), self.assertRaises(RuntimeError) as ctx:
                resolve_device("cuda")
            self.assertIn("cuda", str(ctx.exception).lower())

    def test_explicit_mps_request_raises_when_unavailable(self):
        with _Availability(cuda=True, mps=False), self.assertRaises(RuntimeError) as ctx:
            resolve_device("mps")
        self.assertIn("mps", str(ctx.exception).lower())

    def test_auto_never_raises_whatever_is_available(self):
        """auto is the only mode allowed to degrade, so it must always return something usable."""
        for cuda, mps in ((True, True), (True, False), (False, True), (False, False)):
            with _Availability(cuda=cuda, mps=mps):
                self.assertIn(resolve_device("auto").type, ("cuda", "mps", "cpu"))

    def test_use_cpu_wins_over_an_unavailable_explicit_request(self):
        """use_cpu=True is an explicit CPU instruction, so it must not trip the availability check."""
        with _Availability(cuda=False, mps=False):
            self.assertEqual(resolve_device("cuda", use_cpu=True).type, "cpu")
            self.assertEqual(resolve_device("mps", use_cpu=True).type, "cpu")

    def test_none_is_treated_as_auto(self):
        with _Availability(cuda=False, mps=True):
            self.assertEqual(resolve_device(None).type, "mps")

    def test_accepts_torch_device_and_is_case_insensitive(self):
        with _Availability(cuda=False, mps=True):
            self.assertEqual(resolve_device(torch.device("mps")).type, "mps")
            self.assertEqual(resolve_device("MPS").type, "mps")
            self.assertEqual(resolve_device(" mps ").type, "mps")

    def test_unknown_device_raises(self):
        for bad in ("gpu", "metal", "tpu", ""):
            with self.assertRaises(ValueError):
                resolve_device(bad)

    def test_resolving_mps_enables_cpu_fallback(self):
        previous = os.environ.pop(MPS_FALLBACK_ENV, None)
        try:
            with _Availability(cuda=False, mps=True):
                resolve_device("mps")
            self.assertEqual(os.environ.get(MPS_FALLBACK_ENV), "1")
        finally:
            os.environ.pop(MPS_FALLBACK_ENV, None)
            if previous is not None:
                os.environ[MPS_FALLBACK_ENV] = previous

    def test_cpu_resolution_does_not_touch_fallback_env(self):
        previous = os.environ.pop(MPS_FALLBACK_ENV, None)
        try:
            with _Availability(cuda=False, mps=False):
                resolve_device("cpu")
            self.assertNotIn(MPS_FALLBACK_ENV, os.environ)
        finally:
            if previous is not None:
                os.environ[MPS_FALLBACK_ENV] = previous


class Test_EnableMpsFallback(unittest.TestCase):
    def test_does_not_override_user_setting(self):
        previous = os.environ.get(MPS_FALLBACK_ENV)
        try:
            os.environ[MPS_FALLBACK_ENV] = "0"
            enable_mps_cpu_fallback()
            self.assertEqual(os.environ[MPS_FALLBACK_ENV], "0")
        finally:
            os.environ.pop(MPS_FALLBACK_ENV, None)
            if previous is not None:
                os.environ[MPS_FALLBACK_ENV] = previous


class Test_DeviceToDdevice(unittest.TestCase):
    def test_maps_each_backend_to_its_string(self):
        self.assertEqual(device_to_ddevice(torch.device("cpu")), "cpu")
        self.assertEqual(device_to_ddevice(torch.device("mps")), "mps")
        self.assertEqual(device_to_ddevice(torch.device("cuda", 0)), "cuda")

    def test_index_is_dropped(self):
        self.assertEqual(device_to_ddevice(torch.device("cuda", 3)), "cuda")

    def test_accepts_strings(self):
        self.assertEqual(device_to_ddevice("mps"), "mps")

    def test_unsupported_backend_raises(self):
        with self.assertRaises(ValueError):
            device_to_ddevice(torch.device("meta"))


if __name__ == "__main__":
    unittest.main()
