"""Wider scripted spawn: flag-off poses, holdouts, and the leak check."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mimic_arm.cube_pose import (
    OUTSIDE_X_RANGE,
    OUTSIDE_Y_RANGE,
    in_default_box,
    resolve_cube_sampler,
    sample_transfer_cube_pose,
)
from mimic_arm.saving import step_dirname
from mimic_arm.wider_spawn import (
    FAR_Y_CORNER_X,
    FAR_Y_CORNER_Y,
    GRID_STEP_M,
    HOLDOUT_FRACTION,
    HOLDOUT_SEED,
    HOLDOUTS_PATH,
    LEAK_TOLERANCE_M,
    SHIFT_MAX_M,
    SHIFTS_PER_SPOT,
    assign_holdouts,
    candidate_spots,
    demo_cube_pose,
    holdout_spots,
    in_far_y_corner,
    leak_pairs,
    load_spawn_plan,
    pose_from_xy,
    sample_box_pose_gym,
    shift_distance_ok,
    wider_training_poses,
)
from evaluate import resolve_episode_count
from record_scripted_demos import build_parser, episode_count


class WiderSpawnTest(unittest.TestCase):
    def test_flag_defaults_off(self) -> None:
        args = build_parser().parse_args([])
        self.assertFalse(args.wider_spawn)
        self.assertFalse(args.write_holdouts)
        self.assertFalse(args.write_shifts)
        self.assertIsNone(args.episodes)
        self.assertEqual(args.seed, 0)

    def test_flag_off_matches_gym_sample_box_pose(self) -> None:
        from gym_aloha.utils import sample_box_pose

        def _boom(*_args, **_kwargs):
            raise AssertionError("flag-off recording must not read the holdout file")

        import mimic_arm.wider_spawn as wider_spawn

        original = wider_spawn.load_spawn_plan
        wider_spawn.load_spawn_plan = _boom
        try:
            for episode in range(20):
                pose = demo_cube_pose(episode, wider_spawn=False, seed=0)
                expected = sample_box_pose_gym(episode)
                gym_pose = sample_box_pose(episode)
                np.testing.assert_allclose(pose, expected)
                np.testing.assert_allclose(pose, gym_pose)
                self.assertTrue(in_default_box(float(pose[0]), float(pose[1])))
        finally:
            wider_spawn.load_spawn_plan = original

    def test_flag_off_seed_shifts_the_episode(self) -> None:
        pose = demo_cube_pose(3, wider_spawn=False, seed=10)
        np.testing.assert_allclose(pose, sample_box_pose_gym(13))

    def test_grid_covers_the_outside_rectangle(self) -> None:
        spots = candidate_spots()
        self.assertEqual(len(spots), 16 * 16)
        self.assertEqual(spots[0], (OUTSIDE_X_RANGE[0], OUTSIDE_Y_RANGE[0]))
        self.assertEqual(spots[-1], (OUTSIDE_X_RANGE[1], OUTSIDE_Y_RANGE[1]))
        self.assertEqual(GRID_STEP_M, 0.02)

    def test_far_y_corner_bounds(self) -> None:
        self.assertEqual(FAR_Y_CORNER_X, (0.2, 0.25))
        self.assertEqual(FAR_Y_CORNER_Y, (0.6, 0.65))
        self.assertTrue(in_far_y_corner(0.21, 0.61))
        self.assertTrue(in_far_y_corner(0.25, 0.65))
        self.assertFalse(in_far_y_corner(0.2, 0.65))
        self.assertFalse(in_far_y_corner(0.25, 0.6))
        self.assertFalse(in_far_y_corner(-0.05, 0.65))
        corner = [spot for spot in candidate_spots() if in_far_y_corner(*spot)]
        self.assertEqual(
            corner,
            [
                (0.21, 0.61),
                (0.21, 0.63),
                (0.21, 0.65),
                (0.23, 0.61),
                (0.23, 0.63),
                (0.23, 0.65),
                (0.25, 0.61),
                (0.25, 0.63),
                (0.25, 0.65),
            ],
        )

    def test_assign_holdouts_keeps_the_lists_apart(self) -> None:
        reachable = [spot for spot in candidate_spots() if spot[0] <= 0.1]
        assigned = assign_holdouts(reachable, fraction=HOLDOUT_FRACTION, seed=HOLDOUT_SEED)
        scattered = set(assigned["scattered"])
        corner = set(assigned["far_y_corner"])
        training = set(assigned["training"])
        self.assertEqual(len(scattered), int(round(HOLDOUT_FRACTION * len(reachable))))
        self.assertTrue(scattered.isdisjoint(training))
        self.assertTrue(corner.isdisjoint(training))
        self.assertTrue(scattered.isdisjoint(corner))
        self.assertTrue(scattered <= set(assigned["reachable"]))
        self.assertEqual(assigned["scattered"], sorted(scattered))
        again = assign_holdouts(reachable, fraction=HOLDOUT_FRACTION, seed=HOLDOUT_SEED)
        self.assertEqual(again["scattered"], assigned["scattered"])
        self.assertEqual(again["training"], assigned["training"])

    def test_checked_in_plan_matches_the_assignment(self) -> None:
        plan = load_spawn_plan(HOLDOUTS_PATH)
        assigned = assign_holdouts(list(plan.reachable), fraction=HOLDOUT_FRACTION, seed=HOLDOUT_SEED)
        self.assertEqual(list(plan.scattered), assigned["scattered"])
        self.assertEqual(list(plan.training), assigned["training"])
        self.assertEqual(list(plan.far_y_corner), assigned["far_y_corner"])
        candidates = set(candidate_spots())
        reachable = set(plan.reachable)
        rejected = set(plan.rejected)
        self.assertTrue(reachable.isdisjoint(rejected))
        self.assertEqual(reachable | rejected, candidates)
        self.assertEqual(len(reachable) + len(rejected), len(candidates))
        self.assertEqual(plan.raw["seed"], HOLDOUT_SEED)
        self.assertEqual(plan.raw["holdout_fraction"], HOLDOUT_FRACTION)
        self.assertEqual(plan.raw["candidate_count"], 256)
        self.assertEqual(plan.raw["reachable_count"], 242)
        self.assertEqual(plan.raw["rejected_count"], 14)
        self.assertEqual(plan.raw["joint_only_count"], 0)
        self.assertEqual(plan.raw["in_box_candidate_count"], 100)
        self.assertEqual(plan.raw["in_box_reachable_count"], 100)
        self.assertEqual(plan.raw["outside_band_reachable_count"], 142)
        self.assertEqual(plan.raw["scattered_count"], 36)
        self.assertEqual(plan.raw["far_y_corner_count"], 3)
        self.assertEqual(plan.raw["far_y_corner_unreachable_count"], 6)
        self.assertEqual(plan.raw["training_count"], 203)
        self.assertEqual(
            [tuple(spot) for spot in plan.raw["far_y_corner"]],
            [(0.21, 0.61), (0.21, 0.63), (0.23, 0.61)],
        )
        self.assertEqual(plan.raw["wider_x"], list(OUTSIDE_X_RANGE))
        self.assertEqual(plan.raw["wider_y"], list(OUTSIDE_Y_RANGE))
        self.assertGreater(plan.raw["rejected_count"], 0)
        reachable_corner = [spot for spot in plan.reachable if spot in set(plan.far_y_corner)]
        self.assertEqual(
            len(plan.training) + len(plan.scattered) + len(reachable_corner),
            len(plan.reachable),
        )

    def test_no_held_out_spot_leaks_into_training_demos(self) -> None:
        plan = load_spawn_plan(HOLDOUTS_PATH)
        holdouts = list(plan.scattered) + list(plan.far_y_corner)
        holdouts += [tuple(spot) for spot in plan.raw["far_y_corner_unreachable"]]
        training = list(plan.training)
        self.assertGreater(len(training) * len(holdouts), 100)
        self.assertEqual(leak_pairs(training, holdouts, LEAK_TOLERANCE_M), [])
        poses = wider_training_poses()
        pose_xy = [(float(pose[0]), float(pose[1])) for pose in poses]
        self.assertEqual(pose_xy, training)
        self.assertEqual(leak_pairs(pose_xy, holdouts, LEAK_TOLERANCE_M), [])
        distances = [
            float(np.hypot(tx - hx, ty - hy))
            for tx, ty in training
            for hx, hy in holdouts
        ]
        self.assertGreater(min(distances), 0.01)

        shifts = holdout_spots("holdout-scattered")
        self.assertEqual(leak_pairs(training, shifts, LEAK_TOLERANCE_M), [])
        shift_gaps = [
            float(np.hypot(tx - hx, ty - hy))
            for tx, ty in training
            for hx, hy in shifts
        ]
        self.assertGreater(min(shift_gaps), LEAK_TOLERANCE_M)

        leaked = list(training)
        leaked.append(holdouts[0])
        found = leak_pairs(leaked, holdouts, LEAK_TOLERANCE_M)
        self.assertTrue(found)
        self.assertEqual(found[0][2], 0.0)

        one = [holdouts[0]]
        nudged = list(training) + [(holdouts[0][0] + 0.004, holdouts[0][1])]
        self.assertTrue(leak_pairs(nudged, one, LEAK_TOLERANCE_M))
        clear = list(training) + [(holdouts[0][0] + 0.008, holdouts[0][1])]
        self.assertEqual(leak_pairs(clear, one, LEAK_TOLERANCE_M), [])

    def test_wider_recorder_stops_at_the_training_list(self) -> None:
        plan = load_spawn_plan(HOLDOUTS_PATH)
        with self.assertRaises(ValueError):
            demo_cube_pose(len(plan.training), wider_spawn=True, seed=0)
        with self.assertRaises(ValueError):
            episode_count(True, len(plan.training) + 1, len(plan.training))
        self.assertEqual(episode_count(True, None, len(plan.training)), len(plan.training))
        self.assertEqual(episode_count(False, None, len(plan.training)), 50)

    def test_holdout_samplers_walk_their_own_lists(self) -> None:
        plan = load_spawn_plan(HOLDOUTS_PATH)
        scattered = resolve_cube_sampler("holdout-scattered", None, None)
        corner = resolve_cube_sampler("holdout-far-y", None, None)
        assert scattered is not None and corner is not None
        shifts = holdout_spots("holdout-scattered")
        self.assertEqual(len(shifts), scattered.spot_count)
        for index, spot in enumerate(shifts):
            pose = scattered(index)
            self.assertEqual((float(pose[0]), float(pose[1])), spot)
        for seed in range(1000, 1003):
            pose = corner(seed)
            spot = (float(pose[0]), float(pose[1]))
            self.assertEqual(spot, plan.far_y_corner[seed % len(plan.far_y_corner)])
        record = {
            "mode": "holdout-scattered",
            "x": list(scattered.x_range),
            "y": list(scattered.y_range),
            "exclude_default": scattered.exclude_default,
        }
        self.assertEqual(record["exclude_default"], False)
        self.assertEqual(scattered.spot_count, len(holdout_spots("holdout-scattered")))
        self.assertIn("wider_spawn_holdouts.json", scattered.source)

    def test_holdout_ranges_reject_explicit_bounds(self) -> None:
        with self.assertRaises(ValueError):
            resolve_cube_sampler("holdout-scattered", (0.0, 0.2), (0.4, 0.6))
        with self.assertRaises(ValueError):
            resolve_cube_sampler("holdout-far-y", None, (0.4, 0.6))

    def test_default_and_outside_are_unchanged(self) -> None:
        self.assertIsNone(resolve_cube_sampler("default", None, None))
        outside = resolve_cube_sampler("outside", None, None)
        assert outside is not None
        self.assertEqual(outside.x_range, OUTSIDE_X_RANGE)
        self.assertEqual(outside.y_range, OUTSIDE_Y_RANGE)
        self.assertTrue(outside.exclude_default)
        for seed in range(1000, 1030):
            pose = outside(seed)
            direct = sample_transfer_cube_pose(
                seed, OUTSIDE_X_RANGE, OUTSIDE_Y_RANGE, exclude_default=True
            )
            np.testing.assert_array_equal(pose, direct)
            self.assertFalse(in_default_box(float(pose[0]), float(pose[1])))

    def test_shifted_positions_stay_closer_to_their_anchor(self) -> None:
        plan = load_spawn_plan(HOLDOUTS_PATH)
        rows = plan.raw["scattered_shifts"]["positions"]
        self.assertGreaterEqual(len(rows), len(plan.scattered))
        per_anchor: dict[tuple[float, float], int] = {}
        for row in rows:
            point = (float(row["x"]), float(row["y"]))
            anchor = (float(row["anchor"][0]), float(row["anchor"][1]))
            self.assertIn(anchor, set(plan.scattered))
            self.assertTrue(shift_distance_ok(point, anchor, plan.training, max_shift_m=SHIFT_MAX_M))
            dist_own = float(np.hypot(point[0] - anchor[0], point[1] - anchor[1]))
            dist_train = min(
                float(np.hypot(point[0] - tx, point[1] - ty)) for tx, ty in plan.training
            )
            self.assertLess(dist_own, dist_train)
            self.assertLessEqual(dist_own, SHIFT_MAX_M)
            self.assertGreater(dist_train, LEAK_TOLERANCE_M)
            per_anchor[anchor] = per_anchor.get(anchor, 0) + 1
        short = plan.raw["scattered_shifts"]["shortfalls"]
        short_anchors = {tuple(item["anchor"]) for item in short}
        for anchor in plan.scattered:
            got = per_anchor.get(anchor, 0)
            if anchor in short_anchors:
                self.assertLess(got, SHIFTS_PER_SPOT)
            else:
                self.assertEqual(got, SHIFTS_PER_SPOT)

        # A point past 1 cm, or closer to a training spot, is not a valid shift.
        anchor = plan.scattered[0]
        self.assertFalse(
            shift_distance_ok((anchor[0] + 0.012, anchor[1]), anchor, plan.training)
        )
        training_spot = plan.training[0]
        self.assertFalse(
            shift_distance_ok(
                (training_spot[0] + 0.001, training_spot[1]),
                anchor,
                plan.training,
            )
        )

    def test_scattered_eval_defaults_to_one_episode_per_shift(self) -> None:
        count = len(holdout_spots("holdout-scattered"))
        self.assertEqual(resolve_episode_count(None, "holdout-scattered"), count)
        self.assertEqual(resolve_episode_count(None, "holdout-far-y"), 20)
        self.assertEqual(resolve_episode_count(None, "outside"), 20)
        self.assertEqual(resolve_episode_count(50, "holdout-scattered"), 50)

    def test_fifty_k_weights_folder_is_zero_padded(self) -> None:
        self.assertEqual(step_dirname(50_000, 100_000), "050000")
        self.assertEqual(step_dirname(100_000, 100_000), "100000")
        self.assertEqual(step_dirname(50_000, 50_000), "050000")

    def test_in_box_grid_clears_the_sanity_gate(self) -> None:
        try:
            import gym_aloha  # noqa: F401
        except ImportError:
            self.skipTest("gym_aloha is not installed")
        from mimic_arm.scripted_rollout import scripted_success

        inbox = [spot for spot in candidate_spots() if in_default_box(*spot)]
        self.assertEqual(len(inbox), 100)
        passed = sum(1 for x, y in inbox if scripted_success(x, y, stiff_weld=True))
        self.assertGreaterEqual(passed, 95)

    def test_scripted_rollout_on_known_spots(self) -> None:
        try:
            import gym_aloha  # noqa: F401
        except ImportError:
            self.skipTest("gym_aloha is not installed")
        from mimic_arm.scripted_rollout import rollout_scripted, scripted_success

        stock = rollout_scripted(pose_from_xy(0.1, 0.5), stiff_weld=False, render_top=False)
        self.assertEqual(len(stock["frames"]), 400)
        self.assertEqual(stock["frames"][0]["state"].shape, (14,))
        self.assertEqual(stock["frames"][0]["action"].shape, (14,))
        self.assertTrue(scripted_success(-0.05, 0.5, stiff_weld=True))
        self.assertFalse(scripted_success(0.25, 0.65, stiff_weld=True))


if __name__ == "__main__":
    unittest.main()
