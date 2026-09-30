"""MUJOCO_GL defaults for Windows, Linux, and macOS."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mimic_arm.mujoco_gl import configure_mujoco_rendering


class ConfigureMujocoRenderingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = os.environ.get("MUJOCO_GL")
        os.environ.pop("MUJOCO_GL", None)

    def tearDown(self) -> None:
        if self._saved is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = self._saved

    def _choose(self, platform: str, *, nvidia: bool = False, libs: tuple[str, ...] = ()) -> str:
        def loads(soname: str) -> bool:
            return soname in libs

        with (
            mock.patch("mimic_arm.mujoco_gl.sys.platform", platform),
            mock.patch("mimic_arm.mujoco_gl._has_nvidia_gpu", return_value=nvidia),
            mock.patch("mimic_arm.mujoco_gl._library_loads", side_effect=loads),
        ):
            return configure_mujoco_rendering()

    def test_win32_defaults_to_glfw_without_probing_linux_libraries(self) -> None:
        def fail_probe(*_args, **_kwargs):
            raise AssertionError("Windows must not use the Linux library probe")

        with (
            mock.patch("mimic_arm.mujoco_gl.sys.platform", "win32"),
            mock.patch("mimic_arm.mujoco_gl._has_nvidia_gpu", side_effect=fail_probe),
            mock.patch("mimic_arm.mujoco_gl._library_loads", side_effect=fail_probe),
        ):
            chosen = configure_mujoco_rendering()

        self.assertEqual(chosen, "glfw")
        self.assertEqual(os.environ["MUJOCO_GL"], "glfw")

    def test_win32_keeps_explicit_mujoco_gl(self) -> None:
        os.environ["MUJOCO_GL"] = "osmesa"
        self.assertEqual(self._choose("win32", nvidia=True, libs=("libEGL.so.1",)), "osmesa")
        self.assertEqual(os.environ["MUJOCO_GL"], "osmesa")

    def test_linux_prefers_egl_then_osmesa(self) -> None:
        self.assertEqual(self._choose("linux", nvidia=True, libs=("libEGL.so.1",)), "egl")
        os.environ.pop("MUJOCO_GL", None)
        self.assertEqual(self._choose("linux", nvidia=False, libs=("libOSMesa.so.8", "libEGL.so.1")), "osmesa")
        os.environ.pop("MUJOCO_GL", None)
        self.assertEqual(self._choose("linux", nvidia=False, libs=("libEGL.so.1",)), "egl")
        os.environ.pop("MUJOCO_GL", None)
        self.assertEqual(self._choose("linux", nvidia=False, libs=()), "osmesa")

    def test_linux_keeps_explicit_mujoco_gl(self) -> None:
        os.environ["MUJOCO_GL"] = "glfw"
        self.assertEqual(self._choose("linux", nvidia=True, libs=("libEGL.so.1",)), "glfw")
        self.assertEqual(os.environ["MUJOCO_GL"], "glfw")

    def test_darwin_uses_the_linux_probe(self) -> None:
        self.assertEqual(self._choose("darwin", nvidia=True, libs=("libEGL.so.1",)), "egl")
        os.environ.pop("MUJOCO_GL", None)
        self.assertEqual(self._choose("darwin", nvidia=False, libs=("libOSMesa.so.8",)), "osmesa")
        os.environ.pop("MUJOCO_GL", None)
        self.assertEqual(self._choose("darwin", nvidia=False, libs=()), "osmesa")
        self.assertNotEqual(os.environ["MUJOCO_GL"], "glfw")

    def test_darwin_keeps_explicit_mujoco_gl(self) -> None:
        os.environ["MUJOCO_GL"] = "glfw"
        self.assertEqual(self._choose("darwin"), "glfw")
        self.assertEqual(os.environ["MUJOCO_GL"], "glfw")


if __name__ == "__main__":
    unittest.main()
