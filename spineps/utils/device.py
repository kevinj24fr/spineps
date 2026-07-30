"""Compute-device resolution for the CPU, CUDA and Apple Silicon (Metal/MPS) backends.

All device decisions in SPINEPS go through :func:`resolve_device` so that a single policy decides which
backend inference runs on. On Apple Silicon this means Metal is picked automatically instead of falling
back to the CPU, which is roughly an order of magnitude slower for the 3D convolutions that dominate
SPINEPS' runtime.
"""

from __future__ import annotations

import os
from typing import Literal, Union

import torch
from TPTBox import Log_Type, No_Logger

Device_Kind = Literal["auto", "cpu", "cuda", "mps"]
# The device spelling used by TPTBox's nnU-Net inference API ("auto" is resolved before it is passed on).
DDevice = Literal["cpu", "cuda", "mps"]

#: Set for Metal runs so ops without a Metal kernel run on the CPU rather than raising.
MPS_FALLBACK_ENV = "PYTORCH_ENABLE_MPS_FALLBACK"

VALID_DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda", "mps")

_logger = No_Logger()


def cuda_is_available() -> bool:
    """Whether a CUDA GPU can be used.

    Returns:
        bool: True if torch reports a usable CUDA device.
    """
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch built without CUDA can raise here
        return False


def mps_is_available() -> bool:
    """Whether the Apple Silicon Metal (MPS) backend can be used.

    Returns:
        bool: True if this is an arm64 Mac with a Metal-capable torch build.
    """
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    try:
        return bool(backend.is_available())
    except Exception:  # pragma: no cover - defensive, older torch builds
        return False


def enable_mps_cpu_fallback() -> None:
    """Allows ops with no Metal kernel to fall back to the CPU instead of raising.

    Must run before the first MPS op, otherwise torch may already have cached the setting. Does not
    override the variable if the user set it explicitly.
    """
    os.environ.setdefault(MPS_FALLBACK_ENV, "1")


def resolve_device(
    device: Union[Device_Kind, str, torch.device, None] = "auto",
    use_cpu: bool = False,
    logger=None,
) -> torch.device:
    """Resolves a device request into a concrete, available :class:`torch.device`.

    With ``"auto"`` the fastest available backend wins: CUDA, then Metal (MPS), then CPU. A backend that
    was asked for by name but is not available degrades rather than crashing: an unavailable CUDA request
    falls through to Metal or the CPU, and an unavailable Metal request falls back to the CPU. Metal is
    never substituted for an explicit CUDA request or vice versa, since the two are not interchangeable
    for reproducing results.

    Args:
        device (Device_Kind | str | torch.device | None, optional): Requested backend, one of "auto",
            "cpu", "cuda" or "mps". None is treated as "auto". Defaults to "auto".
        use_cpu (bool, optional): Legacy switch that forces the CPU and overrides ``device``.
            Defaults to False.
        logger (optional): Logger used to report fallbacks. Defaults to None (module logger).

    Returns:
        torch.device: An available device. CUDA is returned with an explicit index of 0.

    Raises:
        ValueError: If ``device`` is not one of the supported backends.
    """
    logger = logger if logger is not None else _logger

    requested = device.type if isinstance(device, torch.device) else str(device if device is not None else "auto")
    requested = requested.strip().lower()
    # torch.device("cuda:0").type is already "cuda", but a raw "cuda:0" string needs the index stripped.
    requested = requested.split(":")[0]

    if requested not in VALID_DEVICES:
        raise ValueError(f"Unknown device {device!r}, expected one of {VALID_DEVICES}")

    if use_cpu:
        return torch.device("cpu")

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "mps":
        if mps_is_available():
            enable_mps_cpu_fallback()
            return torch.device("mps")
        logger.print("Metal (mps) was requested but is not available, falling back to cpu", Log_Type.WARNING)
        return torch.device("cpu")

    if requested == "cuda":
        if cuda_is_available():
            return torch.device("cuda", 0)
        if mps_is_available():
            logger.print("cuda was requested but is not available, using Metal (mps) instead", Log_Type.WARNING)
            enable_mps_cpu_fallback()
            return torch.device("mps")
        logger.print("cuda was requested but is not available, falling back to cpu", Log_Type.WARNING)
        return torch.device("cpu")

    # "auto": fastest backend first.
    if cuda_is_available():
        return torch.device("cuda", 0)
    if mps_is_available():
        enable_mps_cpu_fallback()
        return torch.device("mps")
    return torch.device("cpu")


def device_to_ddevice(device: Union[torch.device, str]) -> DDevice:
    """Converts a device into the ``ddevice`` string TPTBox's inference API expects.

    Args:
        device (torch.device | str): Device to convert. Any CUDA index is dropped.

    Returns:
        DDevice: One of "cpu", "cuda" or "mps".

    Raises:
        ValueError: If the device is not one of the backends SPINEPS supports.
    """
    kind = device.type if isinstance(device, torch.device) else str(device).strip().lower().split(":")[0]
    if kind not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Device {device!r} is not supported for inference, expected cpu, cuda or mps")
    return kind  # type: ignore[return-value]


def describe_device(device: Union[torch.device, str]) -> str:
    """Builds a human-readable name for a device, for logging.

    Args:
        device (torch.device | str): Device to describe.

    Returns:
        str: Description such as "Metal (Apple Silicon GPU)" or a CUDA device name.
    """
    kind = device.type if isinstance(device, torch.device) else str(device).strip().lower().split(":")[0]
    if kind == "mps":
        return "Metal (Apple Silicon GPU)"
    if kind == "cuda":
        try:
            return f"CUDA ({torch.cuda.get_device_name(0)})"
        except Exception:  # pragma: no cover - name lookup is best effort
            return "CUDA"
    return "CPU"
