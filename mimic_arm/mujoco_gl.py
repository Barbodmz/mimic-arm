"""Pick a headless MuJoCo renderer before MuJoCo is imported.

gym-aloha renders with MuJoCo. On a machine with a GPU, EGL talks to the
GPU. On a CPU-only Linux machine (this project's smoke test, a Colab
runtime that lost its GPU) OSMesa draws frames in software. On Windows the
default is glfw, because the Linux libraries this probe looks for are not
there and OSMesa does not work.

MuJoCo reads the ``MUJOCO_GL`` environment variable at import time, so
call :func:`configure_mujoco_rendering` before importing ``lerobot`` or
``mujoco``. If you already exported ``MUJOCO_GL``, that choice is kept.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
from pathlib import Path


def _library_loads(soname: str) -> bool:
    """True when the dynamic linker can open this shared library."""
    try:
        ctypes.CDLL(soname)
    except OSError:
        return False
    return True


def _has_nvidia_gpu() -> bool:
    return Path("/dev/nvidia0").exists() or shutil.which("nvidia-smi") is not None and Path(
        "/proc/driver/nvidia/version"
    ).exists()


def configure_mujoco_rendering() -> str:
    """Set ``MUJOCO_GL`` if it is unset. Returns the backend name in use."""
    chosen = os.environ.get("MUJOCO_GL")
    if chosen:
        return chosen

    # Windows has neither libEGL.so.1 nor libOSMesa.so.8, so the Linux probe
    # would select osmesa, which MuJoCo cannot use there. glfw is the
    # backend that works for evaluation. Linux and macOS keep the probe:
    # EGL when a GPU and libEGL are present, otherwise OSMesa.
    if sys.platform == "win32":
        chosen = "glfw"
    elif _has_nvidia_gpu() and _library_loads("libEGL.so.1"):
        chosen = "egl"
    elif _library_loads("libOSMesa.so.8"):
        chosen = "osmesa"
    elif _library_loads("libEGL.so.1"):
        chosen = "egl"
    else:
        chosen = "osmesa"

    os.environ["MUJOCO_GL"] = chosen
    print(f"MuJoCo renderer: MUJOCO_GL={chosen}")
    return chosen
