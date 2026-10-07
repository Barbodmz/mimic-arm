"""Demo picker for issue #12: motion style, seeded random, cube-spot fairness."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mimic_arm.cube_spot import (
    CubeSpot,
    RightGripperKinematics,
    coverage_matched_indices,
    cube_spot_from_top_frame,
    project_table_point,
    recovery_problem,
    select_demo_sets,
    spatial_mismatch,
    unproject_table_pixel,
)
from mimic_arm.episodes import (
    apply_episode_filter,
    format_dataset_episodes_flag,
    parse_episodes_argument,
)
from mimic_arm.motion_style import (
    ARM_INDEX,
    FPS,
    most_alike_indices,
    motion_distance,
    motion_style_features,
    pairwise_distances,
    random_episode_indices,
)

import train

try:
    import draccus
except ImportError:  # pragma: no cover - exercised only when the stack is installed
    draccus = None


@dataclass
class _EpisodeDatasetConfig:
    episodes: list[int] | None = None


@dataclass
class _EpisodeTrainConfig:
    dataset: _EpisodeDatasetConfig


def _grip(fraction: np.ndarray, close_at: float) -> np.ndarray:
    values = np.empty(fraction.shape, dtype=float)
    for i, t in enumerate(fraction):
        if t < 0.08:
            values[i] = 0.0
        elif t < 0.20:
            values[i] = (t - 0.08) / 0.12
        elif t < close_at:
            values[i] = 1.0
        else:
            values[i] = 0.40
    return values


def _episode(n: int, arm_at, close_at: float = 0.45, fps: float = FPS) -> np.ndarray:
    """(n, 14) state. ``arm_at(t_seconds)`` returns 12 arm-joint positions."""
    state = np.zeros((n, 14), dtype=float)
    times = np.arange(n) / fps
    fraction = np.linspace(0.0, 1.0, n)
    arm = np.stack([np.asarray(arm_at(t), dtype=float) for t in times])
    state[:, ARM_INDEX] = arm
    gripper = _grip(fraction, close_at)
    state[:, 6] = gripper
    state[:, 13] = gripper
    return state


class MotionStyleTest(unittest.TestCase):
    def test_geometric_path_is_not_the_primary_score(self) -> None:
        def forward(t):
            return np.full(12, 0.4 * t)

        def opposite(t):
            return -np.full(12, 0.4 * t)

        def shifted(t):
            return np.full(12, 0.4 * t + 2.5)

        same_length = 80
        original = motion_style_features(_episode(same_length, forward))
        flipped = motion_style_features(_episode(same_length, opposite))
        offset = motion_style_features(_episode(same_length, shifted))
        path_distance = max(
            motion_distance(original, flipped),
            motion_distance(original, offset),
        )

        def early_burst(t):
            # Same joint travel as ``forward`` over the episode, packed into the first quarter.
            duration = (same_length - 1) / FPS
            if t < 0.25 * duration:
                return np.full(12, 0.4 * duration * (t / (0.25 * duration)))
            return np.full(12, 0.4 * duration)

        different_speed = motion_style_features(_episode(same_length, early_burst))
        style_distance = motion_distance(original, different_speed)
        self.assertLess(path_distance, 1e-6)
        self.assertGreater(style_distance, 0.5)
        self.assertGreater(style_distance, path_distance * 100)

    def test_duration_is_in_the_score_but_does_not_dominate(self) -> None:
        def arm(t):
            return np.full(12, 0.3 * t)

        short = motion_style_features(_episode(40, arm))
        long = motion_style_features(_episode(160, arm))
        duration_distance = motion_distance(short, long)
        self.assertGreater(duration_distance, 0.0)
        self.assertNotAlmostEqual(short.duration_s, long.duration_s)

        def burst(t):
            duration = (80 - 1) / FPS
            if t < 0.2 * duration:
                return np.full(12, 0.3 * duration * (t / (0.2 * duration)))
            return np.full(12, 0.3 * duration)

        steady = motion_style_features(_episode(80, arm))
        rushed = motion_style_features(_episode(80, burst))
        style_distance = motion_distance(steady, rushed)
        self.assertGreater(style_distance, duration_distance)

    def test_gripper_close_timing_changes_the_score(self) -> None:
        def arm(t):
            return np.full(12, 0.2 * t)

        early = motion_style_features(_episode(100, arm, close_at=0.30))
        late = motion_style_features(_episode(100, arm, close_at=0.80))
        self.assertGreater(motion_distance(early, late), 0.2)
        self.assertLess(early.right_close_fraction, late.right_close_fraction)

    def test_checked_in_lists_passed_the_cube_spot_gate(self) -> None:
        payload = json.loads((ROOT / "demo_sets" / "option4_sets.json").read_text())
        self.assertTrue(payload["fairness_passed"])
        self.assertEqual(payload["fairness_method"], "medoid_neighborhood")
        self.assertEqual(payload["seed"], 12)
        self.assertEqual(len(payload["alike"]), 20)
        self.assertEqual(len(payload["random"]), 20)
        self.assertEqual(payload["cube_spot_sources"], {"gripper_close": 50})
        self.assertTrue((ROOT / "demo_sets" / "option4_alike_vs_random.png").is_file())

    def test_seeded_random_list_is_stable(self) -> None:
        first = random_episode_indices(50, 20, 12)
        second = random_episode_indices(50, 20, 12)
        other = random_episode_indices(50, 20, 99)
        self.assertEqual(
            first,
            (0, 5, 9, 14, 17, 22, 23, 24, 28, 29, 30, 33, 35, 40, 41, 42, 43, 47, 48, 49),
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        self.assertEqual(len(set(first)), 20)
        self.assertNotEqual(first, other)
        self.assertEqual(first, tuple(sorted(first)))

    def test_alike_set_has_the_requested_size_and_is_the_medoid_neighborhood(self) -> None:
        def arm(scale):
            return lambda t: np.full(12, scale * t)

        styles = [
            motion_style_features(_episode(60, arm(0.2))),
            motion_style_features(_episode(60, arm(0.2))),
            motion_style_features(_episode(60, arm(0.2))),
            motion_style_features(_episode(60, arm(0.2), close_at=0.85)),
            motion_style_features(_episode(60, arm(1.4), close_at=0.85)),
        ]
        distances = pairwise_distances(styles)
        medoid, alike = most_alike_indices(distances, 3)
        self.assertEqual(len(alike), 3)
        self.assertIn(medoid, alike)
        self.assertEqual(set(alike), {0, 1, 2})


class CubeSpotFairnessTest(unittest.TestCase):
    def _spot(self, episode: int, x: float, y: float, z: float = 0.06) -> CubeSpot:
        return CubeSpot(episode, x, y, z, frame=10, source="gripper_close")

    def test_missing_spots_fail_closed(self) -> None:
        spots = [self._spot(i, 0.05 + 0.01 * i, 0.45) for i in range(6)]
        spots[2] = None
        reason = recovery_problem(spots)
        self.assertIsNotNone(reason)
        assert reason is not None
        self.assertIn("missing", reason)
        distances = np.zeros((6, 6))
        _medoid, alike, random_set, report = select_demo_sets(
            distances, spots, k=3, seed=1, random_indices=(0, 1, 3)
        )
        self.assertFalse(report.passed)
        self.assertIn("missing", report.reason)
        self.assertIsNone(spots[2])
        self.assertEqual(len(alike), 3)
        self.assertEqual(len(random_set), 3)

    def test_collapsed_or_off_table_spots_fail_closed(self) -> None:
        collapsed = [self._spot(i, 0.10, 0.50) for i in range(8)]
        reason = recovery_problem(collapsed)
        self.assertIsNotNone(reason)
        assert reason is not None
        self.assertIn("spread", reason)

        xs = [0.02, 0.08, 0.14, 0.18, 0.03, 0.10, 0.16, 0.12]
        ys = [0.42, 0.48, 0.55, 0.58, 0.44, 0.52, 0.46, 0.57]
        airborne = [self._spot(i, xs[i], ys[i], z=5.0) for i in range(8)]
        height = recovery_problem(airborne)
        self.assertIsNotNone(height)
        assert height is not None
        self.assertIn("table", height)

        off_box = [self._spot(i, 3.0 + xs[i], 3.0 + (ys[i] - 0.4)) for i in range(8)]
        place = recovery_problem(off_box)
        self.assertIsNotNone(place)
        assert place is not None
        self.assertIn("spawn", place)

    def test_unmatchable_spreads_fail_closed(self) -> None:
        wide = [self._spot(i, 0.02 + 0.04 * i, 0.42 + 0.04 * i) for i in range(4)]
        tight = [self._spot(i, 0.11, 0.51 + 0.001 * i) for i in range(4)]
        reason = spatial_mismatch(tight, wide)
        self.assertIsNotNone(reason)

    def test_coverage_match_prefers_the_medoid_over_copying_random(self) -> None:
        spots = [
            self._spot(0, 0.02, 0.42),
            self._spot(1, 0.16, 0.44),
            self._spot(2, 0.16, 0.44),
        ]
        # Random uses the farther episode in the high-x, low-y quadrant.
        distance = np.array([0.0, 0.2, 3.0])
        matched = coverage_matched_indices(distance, spots, random_indices=(2,), k=1)
        self.assertEqual(matched, (1,))

    def test_coverage_match_can_repair_a_clumped_neighborhood(self) -> None:
        spots = [
            self._spot(0, 0.02, 0.42),
            self._spot(1, 0.03, 0.43),
            self._spot(2, 0.02, 0.44),
            self._spot(3, 0.04, 0.42),
            self._spot(4, 0.16, 0.44),
            self._spot(5, 0.04, 0.56),
            self._spot(6, 0.16, 0.56),
            self._spot(7, 0.05, 0.45),
        ]
        motion = np.array([0.0, 0.0, 0.0, 0.0, 1.5, 1.5, 1.5, 3.0])
        distances = np.abs(motion[:, None] - motion[None, :])
        medoid, alike, _random_set, report = select_demo_sets(
            distances,
            spots,
            k=4,
            seed=12,
            random_indices=(7, 4, 5, 6),
        )
        self.assertEqual(medoid, 0)
        self.assertTrue(report.passed)
        self.assertEqual(report.method, "coverage_matched")
        self.assertEqual(set(alike), {0, 4, 5, 6})
        self.assertNotEqual(set(alike), {7, 4, 5, 6})

    def test_medoid_neighborhood_passes_when_spreads_already_match(self) -> None:
        spots = [
            self._spot(0, 0.02, 0.42),
            self._spot(1, 0.18, 0.42),
            self._spot(2, 0.02, 0.58),
            self._spot(3, 0.18, 0.58),
            self._spot(4, 0.04, 0.44),
            self._spot(5, 0.16, 0.44),
            self._spot(6, 0.04, 0.56),
            self._spot(7, 0.16, 0.56),
        ]
        motion = np.array([0.0, 0.1, 0.1, 0.1, 5.0, 5.0, 5.0, 5.0])
        distances = np.abs(motion[:, None] - motion[None, :])
        _medoid, alike, random_set, report = select_demo_sets(
            distances,
            spots,
            k=4,
            seed=12,
            random_indices=(4, 5, 6, 7),
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.method, "medoid_neighborhood")
        self.assertEqual(set(alike), {0, 1, 2, 3})
        self.assertEqual(set(random_set), {4, 5, 6, 7})

    def test_top_camera_roundtrip_and_rejects_a_frame_with_no_cube(self) -> None:
        column, row = project_table_point(0.12, 0.48)
        x, y = unproject_table_pixel(column, row)
        self.assertAlmostEqual(x, 0.12, places=5)
        self.assertAlmostEqual(y, 0.48, places=5)

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        c0, r0 = int(round(column)), int(round(row))
        image[r0 - 3 : r0 + 4, c0 - 3 : c0 + 4, 0] = 220
        spot = cube_spot_from_top_frame(image, episode=3)
        self.assertIsNotNone(spot)
        assert spot is not None
        self.assertEqual(spot.source, "top_camera")
        self.assertAlmostEqual(spot.x, 0.12, delta=0.01)
        self.assertAlmostEqual(spot.y, 0.48, delta=0.01)
        self.assertIsNone(cube_spot_from_top_frame(np.zeros_like(image), episode=3))

    def test_right_gripper_spot_ignores_the_left_arm(self) -> None:
        kin = RightGripperKinematics()
        state = np.zeros(14)
        state[7:13] = [0.1, -0.5, 0.8, 0.0, -0.4, 0.2]
        state[13] = 0.4
        original = kin.cube_xyz(state)
        moved_left = state.copy()
        moved_left[0:6] = 1.0
        np.testing.assert_allclose(kin.cube_xyz(moved_left), original)
        moved_right = state.copy()
        moved_right[8] += 0.4
        self.assertGreater(float(np.linalg.norm(kin.cube_xyz(moved_right) - original)), 1e-3)


class TrainEpisodesFlagTest(unittest.TestCase):
    def test_flag_uses_lerobot_episode_filter_and_not_a_stats_recompute(self) -> None:
        flag = format_dataset_episodes_flag([3, 1, 4])
        self.assertEqual(flag, "--dataset.episodes=[3,1,4]")
        command = apply_episode_filter(["--policy.type=act"], [], [3, 1, 4])
        self.assertEqual(command[-1], flag)
        self.assertFalse(any("stat" in token.lower() for token in command))
        with self.assertRaises(SystemExit):
            apply_episode_filter([], ["--dataset.episodes=[0]"], [1])

    def test_parser_accepts_a_list_and_a_json_file(self) -> None:
        parser = train.build_parser()
        args, extra = parser.parse_known_args(["--episodes", "1,2,3", "--save_freq=10"])
        self.assertEqual(args.episodes, "1,2,3")
        self.assertEqual(extra, ["--save_freq=10"])
        self.assertEqual(parse_episodes_argument("1, 2, 3", None), [1, 2, 3])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sets.json"
            path.write_text(json.dumps({"alike": [1, 2], "random": [3, 4]}))
            self.assertEqual(parse_episodes_argument(str(path), "alike"), [1, 2])
            self.assertEqual(parse_episodes_argument(str(path), "random"), [3, 4])
            with self.assertRaises(SystemExit):
                parse_episodes_argument(str(path), None)

    def test_draccus_decodes_the_flag_when_available(self) -> None:
        if draccus is None:
            self.skipTest("draccus is not installed")
        flag = format_dataset_episodes_flag([0, 5, 9])
        parsed = draccus.parse(_EpisodeTrainConfig, args=[flag])
        self.assertEqual(parsed.dataset.episodes, [0, 5, 9])


if __name__ == "__main__":
    unittest.main()
