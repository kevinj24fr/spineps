# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
"""Guards the numerical consequence of device selection, not just which device gets picked.

The risk this feature introduces is that Metal and the CPU disagree. test_device.py covers the branching
logic with mocked availability; this file checks the thing that actually matters, by running the same real
network on both backends and requiring the outputs to agree.

Skipped unless Metal is available and the model weights are present, so it is a local guard rather than a
CI gate. Run it with SPINEPS_SEGMENTOR_MODELS pointing at your weights directory.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np
import torch

# Floors, not targets. Seeded from measurements on real sagittal T2w data, where the semantic mask scored
# a mean Dice of 0.966 and the instance mask 0.972 between the two backends, with 99.7% of voxels
# identical. These thresholds sit below that so ordinary float jitter passes, but a genuine regression
# (a wrong device, a broken kernel, an fp16 downgrade) does not.
MIN_ARGMAX_AGREEMENT = 0.99
MIN_LOGIT_CORRELATION = 0.999
MAX_MEAN_ABS_LOGIT_DIFF = 1e-3


def _models_dir() -> Path | None:
    """Returns the configured model weights directory, or None if it is not usable."""
    raw = os.environ.get("SPINEPS_SEGMENTOR_MODELS")
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_dir() else None


def _find_instance_model() -> Path | None:
    """Locates a downloaded instance (Unet3D) model folder, which is the cheapest real network to run."""
    root = _models_dir()
    if root is None:
        return None
    for candidate in sorted(root.glob("*instance*")) + sorted(root.glob("*Inst*")):
        if (candidate / "inference_config.json").is_file():
            return candidate
    return None


@unittest.skipUnless(torch.backends.mps.is_available(), "no Metal device on this host")
class Test_Metal_Agrees_With_CPU(unittest.TestCase):
    """Runs one real network on both backends and requires them to agree."""

    def setUp(self):
        self.model_path = _find_instance_model()
        if self.model_path is None:
            self.skipTest("no instance model weights found; set SPINEPS_SEGMENTOR_MODELS")

    def test_metal_and_cpu_agree_on_the_same_input(self):
        from spineps.get_models import get_actual_model

        cpu_model = get_actual_model(self.model_path, device="cpu").load()
        mps_model = get_actual_model(self.model_path, device="mps").load()
        self.assertEqual(cpu_model.device.type, "cpu")
        self.assertEqual(mps_model.device.type, "mps")

        channels = cpu_model.predictor.network.channels
        # Spatial dims must be divisible by 8 for this architecture.
        torch.manual_seed(0)
        x = torch.randn(1, channels, 64, 64, 32)

        with torch.no_grad():
            cpu_logits = cpu_model.predictor(x.clone()).float().cpu()
            mps_logits = mps_model.predictor(x.clone().to("mps")).float().cpu()
        torch.mps.synchronize()

        self.assertEqual(cpu_logits.shape, mps_logits.shape)

        agreement = (cpu_logits.argmax(1) == mps_logits.argmax(1)).float().mean().item()
        mean_abs_diff = (cpu_logits - mps_logits).abs().mean().item()
        correlation = float(np.corrcoef(cpu_logits.flatten().numpy(), mps_logits.flatten().numpy())[0, 1])

        self.assertGreaterEqual(
            agreement,
            MIN_ARGMAX_AGREEMENT,
            f"Metal and CPU disagree on {(1 - agreement) * 100:.3f}% of voxels, above the tolerated floor",
        )
        self.assertGreaterEqual(correlation, MIN_LOGIT_CORRELATION, "logits are no longer strongly correlated")
        self.assertLessEqual(mean_abs_diff, MAX_MEAN_ABS_LOGIT_DIFF, "mean absolute logit difference grew")

    def test_metal_is_not_silently_running_in_half_precision(self):
        """fp16 would still pass a loose Dice check while quietly degrading precision, so pin the dtype."""
        from spineps.get_models import get_actual_model

        model = get_actual_model(self.model_path, device="mps").load()
        channels = model.predictor.network.channels
        with torch.no_grad():
            out = model.predictor(torch.zeros(1, channels, 32, 32, 32, device="mps"))
        self.assertEqual(out.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
