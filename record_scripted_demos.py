#!/usr/bin/env python
"""Record scripted transfer-cube demonstrations.

With no extra flags this follows gym-aloha's training box and
``sample_box_pose(seed + episode)``, and the mocap weld stays at the stock
setting. ``--wider-spawn`` records the training list in
``demo_sets/wider_spawn_holdouts.json``. The end-effector teacher uses a
higher mocap impedance so the gripper tracks the waypoint; each demo is kept
only when the joint replay succeeds in the stock eval env. Held-out spots are
never recorded.

``--write-holdouts`` rebuilds that JSON (and the spot plot) by rolling the
scripted policy on the 2 cm grid. It does not record a dataset.

Examples
--------
Rebuild the checked-in holdout file::

    python record_scripted_demos.py --write-holdouts

Record the wider training demos::

    python record_scripted_demos.py --wider-spawn --output-dir outputs/data/scripted_wide
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mimic_arm.wider_spawn import (
    HOLDOUTS_PATH,
    PLOT_PATH,
    build_plan_payload,
    demo_cube_pose,
    load_spawn_plan,
    plot_spawn_plan,
    save_spawn_plan,
)

DEFAULT_EPISODES = 50
TASK = "Pick up the cube with the right arm and transfer it to the left arm."
MOTORS = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]
ARTIFACT_PLOT = Path("/opt/cursor/artifacts/wider_spawn_spots.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--wider-spawn",
        action="store_true",
        help=(
            "Record the saved training spots from the wider box. "
            "Off by default, which keeps gym-aloha's box, seeds, and stock weld."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=(
            f"How many episodes to record. Default {DEFAULT_EPISODES} with the flag off, "
            "or every training spot with --wider-spawn. A shorter wider run records "
            "a prefix of that list."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="First episode seed when --wider-spawn is off. Episode i uses seed + i.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="LeRobot dataset directory. Created empty. Default depends on --wider-spawn.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Dataset repo id stored in the metadata. Default local/scripted_wide with --wider-spawn.",
    )
    parser.add_argument(
        "--write-holdouts",
        action="store_true",
        help=f"Survey the wider grid and rewrite {HOLDOUTS_PATH.name}. Does not record demos.",
    )
    return parser


def episode_count(wider_spawn: bool, episodes: int | None, training_count: int) -> int:
    """How many episodes this invocation records."""
    if episodes is not None and episodes < 1:
        raise ValueError("--episodes must be at least 1.")
    if wider_spawn:
        if training_count < 1:
            raise ValueError("The wider training list is empty.")
        if episodes is None:
            return training_count
        if episodes > training_count:
            raise ValueError(
                f"--episodes {episodes} is past the {training_count} training spots. "
                "The wider recorder does not repeat or borrow a held-out spot."
            )
        return episodes
    if episodes is None:
        return DEFAULT_EPISODES
    return episodes


def default_output_dir(wider_spawn: bool) -> Path:
    name = "scripted_wide" if wider_spawn else "scripted"
    return Path("outputs/data") / name


def default_repo_id(wider_spawn: bool) -> str:
    if wider_spawn:
        return "local/scripted_wide"
    return "local/aloha_sim_transfer_cube_scripted"


def dataset_features() -> dict:
    """User features for the published scripted transfer-cube dataset."""
    names = {"motors": list(MOTORS)}
    vector = {"dtype": "float32", "shape": (14,), "names": names}
    return {
        "observation.images.top": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": dict(vector),
        "action": {"dtype": "float32", "shape": (14,), "names": names},
    }


def write_holdouts() -> dict:
    """Roll the scripted policy on the wider grid and save the holdout file."""
    from mimic_arm.mujoco_gl import configure_mujoco_rendering
    from mimic_arm.scripted_rollout import survey_reachable_spots

    configure_mujoco_rendering()
    print("Surveying the wider grid. Each spot is one scripted handover.", flush=True)
    reachable = survey_reachable_spots()
    payload = build_plan_payload(
        reachable,
        joint_only_count=survey_reachable_spots.last_joint_only,
    )
    path = save_spawn_plan(payload)
    plan = load_spawn_plan(path)
    plot_spawn_plan(plan, PLOT_PATH)
    try:
        plot_spawn_plan(plan, ARTIFACT_PLOT)
    except OSError as exc:
        print(f"Artifact plot skipped: {exc}")
    print(f"Wrote {path}")
    print(f"Wrote {PLOT_PATH}")
    print(
        f"candidates {payload['candidate_count']}  "
        f"reachable {payload['reachable_count']}  "
        f"rejected {payload['rejected_count']}  "
        f"outside reachable {payload['outside_band_reachable_count']}  "
        f"in-box {payload['in_box_reachable_count']}/{payload['in_box_candidate_count']}  "
        f"scattered {payload['scattered_count']}  "
        f"far-Y {payload['far_y_corner_count']}  "
        f"far-Y unreachable {payload['far_y_corner_unreachable_count']}  "
        f"training {payload['training_count']}"
    )
    return payload


def _write_positions(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, indent=2) + "\n")


def record_demos(
    *,
    wider_spawn: bool,
    episodes: int | None,
    seed: int,
    output_dir: Path,
    repo_id: str,
) -> Path:
    """Write a LeRobot dataset. Each episode is saved before the next one starts."""
    from mimic_arm.mujoco_gl import configure_mujoco_rendering

    configure_mujoco_rendering()
    training_count = 0
    if wider_spawn:
        training_count = len(load_spawn_plan().training)
    count = episode_count(wider_spawn, episodes, training_count)
    if output_dir.exists():
        raise SystemExit(
            f"{output_dir} already exists. Pick a new --output-dir so an earlier dataset stays put."
        )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from mimic_arm.scripted_rollout import rollout_scripted

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        features=dataset_features(),
        root=output_dir,
        robot_type="aloha",
        use_videos=True,
    )
    positions_path = output_dir / "cube_positions.json"
    positions: list[dict] = []
    print(
        f"Recording {count} episodes  wider_spawn={wider_spawn}  "
        f"repo_id={repo_id}  output={output_dir}"
    )
    try:
        for index in range(count):
            pose = demo_cube_pose(index, wider_spawn=wider_spawn, seed=seed)
            result = rollout_scripted(pose, stiff_weld=wider_spawn, render_top=True)
            if wider_spawn and not result["success"]:
                x, y = float(pose[0]), float(pose[1])
                raise RuntimeError(
                    f"Training spot ({x}, {y}) did not finish the handover "
                    f"(joint reward {result['joint_reward']}). "
                    "It was left out of the dataset."
                )
            frames = result["frames"]
            if len(frames) != 400:
                raise RuntimeError(f"Episode {index} has {len(frames)} frames; expected 400.")
            for frame in frames:
                image = frame["image"]
                if getattr(image, "shape", None) != (480, 640, 3):
                    raise RuntimeError(f"Top camera frame shape is {getattr(image, 'shape', None)}.")
                dataset.add_frame(
                    {
                        "observation.images.top": image,
                        "observation.state": frame["state"],
                        "action": frame["action"],
                        "task": TASK,
                    }
                )
            dataset.save_episode()
            positions.append(
                {
                    "episode": index,
                    "seed": None if wider_spawn else int(seed) + index,
                    "x": float(pose[0]),
                    "y": float(pose[1]),
                    "joint_reward": int(result["joint_reward"]),
                    "success": bool(result["success"]),
                }
            )
            _write_positions(positions_path, positions)
            print(
                f"episode {index:03d}  cube ({pose[0]:.3f}, {pose[1]:.3f})  "
                f"reward {result['joint_reward']}",
                flush=True,
            )
    finally:
        dataset.finalize()
    print(f"Wrote {count} episodes to {output_dir}")
    return output_dir


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.write_holdouts:
        if args.wider_spawn or args.episodes is not None:
            raise SystemExit("--write-holdouts only rebuilds the holdout file.")
        write_holdouts()
        return
    output_dir = args.output_dir or default_output_dir(args.wider_spawn)
    repo_id = args.repo_id or default_repo_id(args.wider_spawn)
    record_demos(
        wider_spawn=args.wider_spawn,
        episodes=args.episodes,
        seed=args.seed,
        output_dir=output_dir,
        repo_id=repo_id,
    )


if __name__ == "__main__":
    main()
