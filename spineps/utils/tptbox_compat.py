"""Compatibility shims for upstream TPTBox behaviour that SPINEPS depends on.

TPTBox's nnU-Net sliding-window predictor asks CUDA how much GPU memory is free before it starts, in order
to decide whether a volume has to be split into chunks and whether it should wait for a busy GPU. Up to and
including TPTBox 0.7.6 those queries call ``torch.cuda.mem_get_info`` unconditionally, which raises
``ValueError: Expected a cuda device, but got: mps`` (or ``cpu``) on any machine without CUDA. That makes
every nnU-Net phase fail on Apple Silicon and on CPU-only hosts.

:func:`patch_nnunet_gpu_memory_helpers` replaces those two module-level helpers with device-aware versions.
CUDA keeps using TPTBox's original implementation, so behaviour on CUDA hosts is unchanged.
"""

from __future__ import annotations

import os

import torch

_patched = False


def _mps_free_memory_mb() -> float:
    """Free memory available to Metal, in MB.

    Returns:
        float: Recommended Metal working-set size minus what the driver already holds.
    """
    recommended = float(torch.mps.recommended_max_memory())
    allocated = float(torch.mps.driver_allocated_memory())
    return max(recommended - allocated, 0.0) / 1024**2


def _mps_utilisation() -> float:
    """Fraction of the Metal working set currently in use, in [0, 1].

    Returns:
        float: 0.0 when nothing is allocated, approaching 1.0 as the working set fills.
    """
    recommended = float(torch.mps.recommended_max_memory())
    if recommended <= 0:
        return 0.0
    return min(max(float(torch.mps.driver_allocated_memory()) / recommended, 0.0), 1.0)


def _host_free_memory_mb() -> float:
    """Free system RAM in MB, falling back to total RAM when the OS will not report free pages.

    Returns:
        float: Best available estimate of usable host memory in MB.
    """
    page_size = os.sysconf("SC_PAGE_SIZE")
    for key in ("SC_AVPHYS_PAGES", "SC_PHYS_PAGES"):
        try:
            pages = os.sysconf(key)
        except (ValueError, OSError, AttributeError):  # pragma: no cover - platform dependent
            continue
        if pages and pages > 0:
            return float(pages) * float(page_size) / 1024**2
    return 0.0  # pragma: no cover - no sysconf memory information at all


def _device_kind(device) -> str:
    """Normalizes a device argument to its backend name.

    Args:
        device: A torch.device, device string, or index.

    Returns:
        str: The backend name, e.g. "cuda", "mps" or "cpu".
    """
    if isinstance(device, torch.device):
        return device.type
    if device is None:
        return "cuda"  # TPTBox's own default when no device is given
    if isinstance(device, int):
        return "cuda"
    return str(device).strip().lower().split(":")[0]


def patch_nnunet_gpu_memory_helpers() -> bool:
    """Makes TPTBox's nnU-Net GPU-memory queries work on Metal and CPU.

    Wraps ``get_gpu_memory_MB`` and ``get_gpu_util`` in TPTBox's predictor module so that non-CUDA devices
    report their own memory instead of raising. CUDA requests are delegated to the original functions
    unchanged. Safe to call repeatedly; only the first call patches.

    Returns:
        bool: True if the patch was applied (or was already in place), False if TPTBox's predictor module
            does not expose the helpers, e.g. because upstream restructured or fixed them.
    """
    global _patched  # noqa: PLW0603
    if _patched:
        return True

    try:
        from TPTBox.segmentation.nnUnet_utils import predictor as tptbox_predictor
    except ImportError:  # pragma: no cover - TPTBox is a hard dependency
        return False

    original_memory = getattr(tptbox_predictor, "get_gpu_memory_MB", None)
    original_util = getattr(tptbox_predictor, "get_gpu_util", None)
    if original_memory is None or original_util is None:
        return False

    def get_gpu_memory_MB(device) -> float:  # name must match the TPTBox helper being replaced
        """Free memory in MB for CUDA, Metal or the host, depending on the device."""
        kind = _device_kind(device)
        if kind == "cuda":
            return original_memory(device)
        if kind == "mps":
            return _mps_free_memory_mb()
        return _host_free_memory_mb()

    def get_gpu_util(device) -> float:
        """Memory utilisation in [0, 1] for CUDA or Metal; 0.0 on CPU, which never has to wait."""
        kind = _device_kind(device)
        if kind == "cuda":
            return original_util(device)
        if kind == "mps":
            return _mps_utilisation()
        return 0.0

    tptbox_predictor.get_gpu_memory_MB = get_gpu_memory_MB
    tptbox_predictor.get_gpu_util = get_gpu_util
    _patched = True
    return True
