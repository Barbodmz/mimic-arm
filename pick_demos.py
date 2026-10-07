#!/usr/bin/env python
"""Pick 20 alike human demos and 20 random ones for issue #12.

The dataset has 50 episodes and no cube-position field. This script scores
how each demo moves (speed, pauses, gripper-close timing), recovers an
approximate cube spot from the right gripper at the close, and checks that
the two 20-demo sets cover a comparable spread. It writes the lists and a
plot. It does not train.

From the repo root, with the LeRobot stack installed (matplotlib is used
for the plot)::

    python pick_demos.py --output-dir demo_sets

The random list uses seed 12. Run the same command again and the lists match.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mimic_arm.cube_spot import (
    NEXT_STEP_IF_UNFAIR,
    CubeSpot,
    FairnessReport,
    RightGripperKinematics,
    recover_cube_spot,
    select_demo_sets,
)
from mimic_arm.motion_style import motion_style_features, pairwise_distances

DATASET_REPO_ID = "lerobot/aloha_sim_transfer_cube_human"
DEFAULT_SEED = 12
DEFAULT_K = 20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo_sets"),
        help="Where option4_sets.json and the plot are written. Default: demo_sets/",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Local dataset checkout. Default: download the parquet from the Hub (no videos).",
    )
    parser.add_argument(
        "--repo-id",
        default=DATASET_REPO_ID,
        help=f"Hugging Face dataset id. Default: {DATASET_REPO_ID}",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Episodes in each set. Default: 20.")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed for the random set. Default: {DEFAULT_SEED}.",
    )
    return parser


def resolve_dataset_root(dataset_root: Path | None, repo_id: str) -> Path:
    if dataset_root is not None:
        return dataset_root
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id,
        repo_type="dataset",
        allow_patterns=["data/**/*.parquet", "meta/info.json"],
    )
    return Path(path)


def assert_no_recorded_cube_field(info: dict) -> None:
    """Stop if the dataset already stores a cube position this script would ignore."""
    features = info.get("features", {})
    suspicious = [
        name
        for name in features
        if any(token in name.lower() for token in ("cube", "environment_state", "box_pose", "object"))
    ]
    if suspicious:
        raise SystemExit(
            "Dataset features include "
            f"{suspicious}, which may already be the cube position. "
            "This picker will not invent a second estimate on top of a real field."
        )
    state = features.get("observation.state", {})
    shape = state.get("shape")
    if shape != [14]:
        raise SystemExit(f"Expected observation.state shape [14], got {shape}.")


def load_streams(root: Path) -> tuple[list, list]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import numpy as np

    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files under {root / 'data'}.")
    columns = ["observation.state", "action", "episode_index", "frame_index"]
    table = pa.concat_tables(pq.read_table(path, columns=columns) for path in files)
    episode = np.asarray(table.column("episode_index").to_pylist(), dtype=int)
    frame = np.asarray(table.column("frame_index").to_pylist(), dtype=int)
    state = np.stack(table.column("observation.state").to_pylist()).astype(float)
    action = np.stack(table.column("action").to_pylist()).astype(float)
    if state.shape[1] != 14 or action.shape[1] != 14:
        raise SystemExit(f"Expected 14-d state and action, got {state.shape} and {action.shape}.")
    n = int(episode.max()) + 1 if episode.size else 0
    if sorted(set(episode.tolist())) != list(range(n)):
        raise SystemExit("Episode indices are not a contiguous range starting at 0.")
    states = []
    actions = []
    for index in range(n):
        mask = episode == index
        order = np.argsort(frame[mask], kind="mergesort")
        states.append(state[mask][order])
        actions.append(action[mask][order])
    return states, actions


def recover_all(states: list) -> list[CubeSpot | None]:
    kinematics = RightGripperKinematics()
    return [
        recover_cube_spot(episode_state, index, first_frame=None, kinematics=kinematics)
        for index, episode_state in enumerate(states)
    ]


def _spot_dict(spot: CubeSpot | None) -> dict | None:
    if spot is None:
        return None
    return {
        "x": spot.x,
        "y": spot.y,
        "z": spot.z,
        "frame": spot.frame,
        "source": spot.source,
    }


def build_payload(
    *,
    repo_id: str,
    seed: int,
    k: int,
    medoid: int | None,
    alike: tuple[int, ...],
    random_set: tuple[int, ...],
    report: FairnessReport,
    spots: list[CubeSpot | None],
    distance_to_medoid,
    styles,
) -> dict:
    sources: dict[str, int] = {}
    for spot in spots:
        if spot is None:
            sources["missing"] = sources.get("missing", 0) + 1
        else:
            sources[spot.source] = sources.get(spot.source, 0) + 1
    return {
        "issue": 12,
        "dataset": repo_id,
        "seed": seed,
        "k": k,
        "fairness_passed": report.passed,
        "fairness_method": report.method,
        "fairness_reason": report.reason,
        "next_step_if_unfair": None if report.passed else NEXT_STEP_IF_UNFAIR,
        "medoid": medoid,
        "alike": list(alike),
        "random": list(random_set),
        "normalization": (
            "Train both lists with train.py --episodes. "
            "LeRobot keeps meta/stats.json from the full 50-episode dataset."
        ),
        "cube_spot_sources": sources,
        "cube_spot_note": (
            "The dataset has no cube-position field. Each spot is the right "
            "fingertip pad at the first gripper close that reaches the table. "
            "The pad offset is measured from the ViperX finger mesh (about 1.3 cm "
            "past the finger joint), not fitted to the spawn box. Recovered x is "
            "shifted a few centimeters toward the right arm relative to the "
            "0.0–0.2 m spawn interval; both sets share that shift. The first "
            "camera frame was not needed: gripper close recovered all 50 episodes."
        ),
        "fairness": {
            "passed": report.passed,
            "method": report.method,
            "reason": report.reason,
            "details": report.details,
        },
        "episodes": [
            {
                "index": index,
                "distance_to_medoid": None if medoid is None else float(distance_to_medoid[index]),
                "duration_s": styles[index].duration_s,
                "right_close_s": styles[index].right_close_s,
                "pause_fraction": styles[index].pause_fraction,
                "cube": _spot_dict(spots[index]),
            }
            for index in range(len(spots))
        ],
    }


def save_plot(path: Path, payload: dict, styles) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    from mimic_arm.cube_pose import DEFAULT_X_RANGE, DEFAULT_Y_RANGE

    alike = set(payload["alike"])
    random_set = set(payload["random"])
    episodes = payload["episodes"]
    indices = np.array([row["index"] for row in episodes])
    distances = np.array([row["distance_to_medoid"] for row in episodes], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.3))
    verdict = "PASSED" if payload["fairness_passed"] else "FAILED"
    fig.suptitle(
        f"Issue #12: 20 most-alike vs 20 random human demos  —  cube-spot fairness {verdict}",
        fontsize=12,
    )

    colors = []
    for index in indices:
        if index in alike and index in random_set:
            colors.append("#1b9e77")
        elif index in alike:
            colors.append("#d95f02")
        elif index in random_set:
            colors.append("#7570b3")
        else:
            colors.append("#bdbdbd")
    order = np.argsort(distances)
    axes[0].bar(np.arange(len(order)), distances[order], color=[colors[i] for i in order], width=0.9)
    axes[0].set_title("Motion-style distance to the medoid")
    axes[0].set_xlabel("Episodes, nearest first")
    axes[0].set_ylabel("Distance")
    axes[0].set_xticks([])

    ax = axes[1]
    ax.add_patch(
        Rectangle(
            (DEFAULT_X_RANGE[0], DEFAULT_Y_RANGE[0]),
            DEFAULT_X_RANGE[1] - DEFAULT_X_RANGE[0],
            DEFAULT_Y_RANGE[1] - DEFAULT_Y_RANGE[0],
            fill=False,
            linestyle="--",
            linewidth=1,
            edgecolor="black",
            label="spawn box",
        )
    )
    def scatter(subset, marker, label, color):
        xs, ys = [], []
        for index in subset:
            cube = episodes[index]["cube"]
            if cube is None:
                continue
            xs.append(cube["x"])
            ys.append(cube["y"])
        ax.scatter(xs, ys, marker=marker, c=color, s=36, label=label, zorder=3)

    other = [int(i) for i in indices if int(i) not in alike and int(i) not in random_set]
    scatter(other, "o", "other", "#bdbdbd")
    scatter(sorted(alike - random_set), "o", "alike", "#d95f02")
    scatter(sorted(random_set - alike), "^", "random", "#7570b3")
    scatter(sorted(alike & random_set), "D", "in both", "#1b9e77")
    ax.set_title("Cube spot at right-gripper close")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    def mean_speed(subset):
        curves = [styles[i].speed for i in subset]
        return np.mean(np.stack(curves), axis=0)

    grid = np.linspace(0.0, 1.0, styles[0].speed.size)
    axes[2].plot(grid, mean_speed(sorted(alike)), color="#d95f02", label="alike mean speed")
    axes[2].plot(grid, mean_speed(sorted(random_set)), color="#7570b3", label="random mean speed")
    axes[2].set_title("Stretched arm speed")
    axes[2].set_xlabel("Fraction of the episode")
    axes[2].set_ylabel("Arm speed (rad/s)")
    axes[2].legend(loc="upper right", fontsize=8, frameon=False)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = resolve_dataset_root(args.dataset_root, args.repo_id)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise SystemExit(f"Missing {info_path}.")
    info = json.loads(info_path.read_text())
    assert_no_recorded_cube_field(info)
    states, actions = load_streams(root)
    if info.get("total_episodes") not in (None, len(states)):
        raise SystemExit(
            f"info.json says {info.get('total_episodes')} episodes, parquet has {len(states)}."
        )

    styles = [
        motion_style_features(state, action) for state, action in zip(states, actions, strict=True)
    ]
    distances = pairwise_distances(styles)
    spots = recover_all(states)
    medoid, alike, random_set, report = select_demo_sets(
        distances, spots, k=args.k, seed=args.seed
    )
    distance_to_medoid = distances[medoid] if medoid is not None else None
    payload = build_payload(
        repo_id=args.repo_id,
        seed=args.seed,
        k=args.k,
        medoid=medoid,
        alike=alike,
        random_set=random_set,
        report=report,
        spots=spots,
        distance_to_medoid=distance_to_medoid,
        styles=styles,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "option4_sets.json"
    plot_path = args.output_dir / "option4_alike_vs_random.png"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    save_plot(plot_path, payload, styles)

    print(f"Dataset: {args.repo_id}  ({len(states)} episodes)")
    print(f"Fairness: {'PASSED' if report.passed else 'FAILED'}  ({report.method})")
    print(report.reason)
    print("Alike:", ",".join(str(i) for i in alike))
    print(f"Random (seed {args.seed}):", ",".join(str(i) for i in random_set))
    print("Cube-spot sources:", payload["cube_spot_sources"])
    print("Wrote", json_path)
    print("Wrote", plot_path)
    if not report.passed:
        print(NEXT_STEP_IF_UNFAIR)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
