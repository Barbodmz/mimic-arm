#!/usr/bin/env python
"""Make ``labmaze==1.0.6`` installable on this interpreter.

``gym-aloha`` depends on ``dm-control``, which depends on ``labmaze``. Upstream
labmaze 1.0.6 publishes binary wheels for CPython 3.7 through 3.12 only. On
Python 3.13, pip falls through to the source distribution, whose build runs
``bazel`` and fails on a fresh Colab (Bazel is not installed, and current
Bazel also ignores the package's WORKSPACE file).

AlohaTransferCube never imports labmaze. When no binary wheel exists, this
installs a pure-Python distribution of the same version so later ``pip
install`` commands treat the requirement as already satisfied and do not
compile the source distribution. On Python 3.12 and older, do not call this:
``requirements.txt`` installs the real wheel.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

LABMAZE_VERSION = "1.0.6"

_PYPROJECT = f"""\
[build-system]
requires = ["setuptools>=71"]
build-backend = "setuptools.build_meta"

[project]
name = "labmaze"
version = "{LABMAZE_VERSION}"
description = "Placeholder for labmaze on interpreters that have no binary wheel."
requires-python = ">=3.9"
"""

_INIT = f'''\
"""Pure-Python stand-in for DeepMind labmaze {LABMAZE_VERSION}.

The maze-generator extensions are not included. gym-aloha's transfer-cube
task does not import this package. Accessing a maze class raises ImportError.
"""

__version__ = "{LABMAZE_VERSION}"


def __getattr__(name):
    raise ImportError(
        f"labmaze.{{name}} is unavailable in this placeholder install. "
        "Upstream labmaze {LABMAZE_VERSION} has no binary wheel for this "
        "Python, and the source build needs Bazel. This project does not "
        "use dm_control locomotion mazes."
    )
'''


def _pip(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pip", *args],
        check=check,
        text=True,
        capture_output=True,
    )


def binary_wheel_available() -> bool:
    """True when PyPI has a labmaze wheel pip can install for this interpreter."""
    result = _pip(
        "install",
        "--dry-run",
        "--ignore-installed",
        "--only-binary=:all:",
        f"labmaze=={LABMAZE_VERSION}",
        check=False,
    )
    if result.returncode == 0:
        return True
    combined = f"{result.stdout}\n{result.stderr}"
    missing = (
        "No matching distribution found" in combined
        or "Could not find a version that satisfies" in combined
    )
    if missing:
        return False
    sys.stderr.write(combined)
    raise SystemExit(
        f"Could not check labmaze wheels (pip exit {result.returncode})."
    )


def install_placeholder() -> None:
    with tempfile.TemporaryDirectory(prefix="labmaze-placeholder-") as tmp:
        root = Path(tmp)
        (root / "pyproject.toml").write_text(_PYPROJECT)
        package = root / "labmaze"
        package.mkdir()
        (package / "__init__.py").write_text(_INIT)
        # Visible on failure: this pip call is not quiet.
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", str(root)]
        )


def main() -> None:
    if sys.version_info < (3, 13):
        print(
            f"Python {sys.version.split()[0]} uses the published labmaze wheel. "
            "Not installing a placeholder."
        )
        return
    if binary_wheel_available():
        print(
            f"labmaze {LABMAZE_VERSION} has a binary wheel for "
            f"Python {sys.version.split()[0]}."
        )
        return
    print(
        f"labmaze {LABMAZE_VERSION} has no binary wheel for "
        f"Python {sys.version.split()[0]}. Installing a pure-Python placeholder "
        "so dm-control does not build the Bazel source distribution. "
        "AlohaTransferCube does not import labmaze.",
        flush=True,
    )
    install_placeholder()


if __name__ == "__main__":
    main()
