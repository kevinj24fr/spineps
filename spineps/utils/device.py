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


def enable_mps_cpu_fallback(logger=None) -> None:
    """Allows ops with no Metal kernel to fall back to the CPU instead of raising.

    Must run before the first MPS op, otherwise torch may already have cached the setting. Does not
    override the variable if the user set it explicitly. Reports the resulting state, because a silent
    fallback shows up only as an unexplained slowdown: any op without a Metal kernel runs on the CPU,
    which costs both time and a device transfer.

    Args:
        logger (optional): Logger used to report the fallback state. Defaults to None (module logger).
    """
    logger = logger if logger is not None else _logger
    if MPS_FALLBACK_ENV in os.environ:
        logger.print(
            f"{MPS_FALLBACK_ENV}={os.environ[MPS_FALLBACK_ENV]} was set externally, leaving it alone",
            Log_Type.NEUTRAL,
        )
        return
    os.environ[MPS_FALLBACK_ENV] = "1"
    logger.print(
        f"{MPS_FALLBACK_ENV}=1: ops without a Metal kernel will run on the CPU instead of failing",
        Log_Type.NEUTRAL,
        verbose=True,
    )


def resolve_device(
    device: Union[Device_Kind, str, torch.device, None] = "auto",
    use_cpu: bool = False,
    logger=None,
) -> torch.device:
    """Resolves a device request into a concrete, available :class:`torch.device`.

    ``"auto"`` picks the fastest available backend: CUDA, then Metal (MPS), then CPU. Naming a backend
    explicitly is a instruction, not a preference: if it is unavailable this raises rather than quietly
    running somewhere else. That matters because the backends are not numerically interchangeable --
    Metal and CPU agree to roughly Dice 0.97 on real data, not exactly -- so a batch script that asks for
    ``cuda`` and silently gets Metal would produce different masks with no error and no record of why.

    Args:
        device (Device_Kind | str | torch.device | None, optional): Requested backend, one of "auto",
            "cpu", "cuda" or "mps". None is treated as "auto". Defaults to "auto".
        use_cpu (bool, optional): Legacy switch that forces the CPU and overrides ``device``.
            Defaults to False.
        logger (optional): Logger used to report which backend "auto" selected. Defaults to None
            (module logger).

    Returns:
        torch.device: An available device. CUDA is returned with an explicit index of 0.

    Raises:
        ValueError: If ``device`` is not one of the supported backends.
        RuntimeError: If a specific backend was requested by name but is not available on this machine.
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

    # An explicitly named backend is honoured or refused; it is never swapped for a different one,
    # because the swap would change the segmentation without saying so.
    if requested == "mps":
        if not mps_is_available():
            raise RuntimeError(
                "device='mps' was requested but Metal is not available on this machine. "
                "On a Mac this usually means the Python interpreter is x86_64 running under Rosetta "
                "instead of native arm64. Use device='auto' to select the best available backend, "
                "or device='cpu' to force the CPU."
            )
        enable_mps_cpu_fallback(logger=logger)
        return torch.device("mps")

    if requested == "cuda":
        if not cuda_is_available():
            raise RuntimeError(
                "device='cuda' was requested but no CUDA GPU is available. Metal is not substituted for "
                "CUDA, since the two do not produce identical segmentations. Use device='auto' to select "
                "the best available backend, or name the backend you want explicitly."
            )
        return torch.device("cuda", 0)

    # "auto": fastest backend first, and say which one was chosen so it ends up in the run log.
    if cuda_is_available():
        return torch.device("cuda", 0)
    if mps_is_available():
        enable_mps_cpu_fallback(logger=logger)
        logger.print("auto-selected Metal (mps); pass -device cpu to force the CPU", Log_Type.NEUTRAL)
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
