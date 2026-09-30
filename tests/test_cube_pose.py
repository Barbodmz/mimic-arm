"""Sampler for AlohaTransferCube cube positions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mimic_arm.cube_pose import (
    CUBE_QUAT,
    CUBE_Z,
    DEFAULT_X_RANGE,
    DEFAULT_Y_RANGE,
    OUTSIDE_X_RANGE,
    OUTSIDE_Y_RANGE,
    in_default_box,
    resolve_cube_sampler,
    sample_transfer_cube_pose,
)


class CubePoseTest(unittest.TestCase):
    def test_installed_sample_box_pose_uses_the_default_box(self) -> None:
        from gym_aloha.utils import sample_box_pose

        for seed in range(30):
            pose = sample_box_pose(seed)
            self.assertTrue(in_default_box(float(pose[0]), float(pose[1])))
            self.assertAlmostEqual(float(pose[2]), CUBE_Z)
            self.assertEqual(tuple(float(v) for v in pose[3:]), CUBE_QUAT)

    def test_default_ranges_match_gym_aloha(self) -> None:
        self.assertEqual(DEFAULT_X_RANGE, (0.0, 0.2))
        self.assertEqual(DEFAULT_Y_RANGE, (0.4, 0.6))
        self.assertEqual(CUBE_Z, 0.05)
        self.assertEqual(OUTSIDE_X_RANGE, (-0.05, 0.25))
        self.assertEqual(OUTSIDE_Y_RANGE, (0.35, 0.65))

    def test_default_box_edges(self) -> None:
        self.assertTrue(in_default_box(0.0, 0.4))
        self.assertTrue(in_default_box(0.2, 0.6))
        self.assertFalse(in_default_box(-0.001, 0.5))
        self.assertFalse(in_default_box(0.201, 0.5))
        self.assertFalse(in_default_box(0.1, 0.399))
        self.assertFalse(in_default_box(0.1, 0.601))

    def test_same_seed_is_same_pose(self) -> None:
        first = sample_transfer_cube_pose(1000, OUTSIDE_X_RANGE, OUTSIDE_Y_RANGE, exclude_default=True)
        second = sample_transfer_cube_pose(1000, OUTSIDE_X_RANGE, OUTSIDE_Y_RANGE, exclude_default=True)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(tuple(first[3:]), CUBE_QUAT)
        self.assertEqual(float(first[2]), CUBE_Z)

    def test_outside_samples_miss_the_training_rectangle(self) -> None:
        for seed in range(1000, 1200):
            pose = sample_transfer_cube_pose(
                seed, OUTSIDE_X_RANGE, OUTSIDE_Y_RANGE, exclude_default=True
            )
            x, y = float(pose[0]), float(pose[1])
            self.assertFalse(in_default_box(x, y), msg=f"seed {seed} landed at {(x, y)}")
            self.assertGreaterEqual(x, OUTSIDE_X_RANGE[0])
            self.assertLessEqual(x, OUTSIDE_X_RANGE[1])
            self.assertGreaterEqual(y, OUTSIDE_Y_RANGE[0])
            self.assertLessEqual(y, OUTSIDE_Y_RANGE[1])

    def test_two_samplers_agree_on_seeds(self) -> None:
        left = resolve_cube_sampler("outside", None, None)
        right = resolve_cube_sampler("outside", None, None)
        assert left is not None and right is not None
        for seed in (1000, 1001, 1002):
            np.testing.assert_array_equal(left(seed), right(seed))
            self.assertEqual(left.positions[seed], right.positions[seed])

    def test_explicit_range_stays_inside_and_can_exclude_default(self) -> None:
        sampler = resolve_cube_sampler("default", (-0.05, 0.0), (0.45, 0.55))
        assert sampler is not None
        pose = sampler(7)
        self.assertLess(float(pose[0]), 0.0)
        self.assertGreaterEqual(float(pose[0]), -0.05)
        self.assertFalse(sampler.exclude_default)

        outside = resolve_cube_sampler("outside", (-0.08, 0.3), (0.3, 0.7))
        assert outside is not None
        for seed in range(50):
            pose = outside(seed)
            self.assertFalse(in_default_box(float(pose[0]), float(pose[1])))

    def test_default_without_bounds_does_not_replace_gym(self) -> None:
        self.assertIsNone(resolve_cube_sampler("default", None, None))

    def test_half_specified_range_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_cube_sampler("default", (0.0, 0.1), None)

    def test_excluding_default_from_an_inside_box_fails(self) -> None:
        with self.assertRaises(RuntimeError):
            sample_transfer_cube_pose(1, (0.05, 0.15), (0.45, 0.55), exclude_default=True)

    def test_numpy_seed_matches_int_seed(self) -> None:
        as_int = sample_transfer_cube_pose(1000, OUTSIDE_X_RANGE, OUTSIDE_Y_RANGE, exclude_default=True)
        as_array = sample_transfer_cube_pose(
            np.array([1000]), OUTSIDE_X_RANGE, OUTSIDE_Y_RANGE, exclude_default=True
        )
        np.testing.assert_array_equal(as_int, as_array)


if __name__ == "__main__":
    unittest.main()
