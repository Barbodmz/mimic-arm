"""Wider cube spawn for scripted demos, with spots held out of training.

The wider rectangle is gym-aloha's training box plus the 5 cm band that
``--cube-range outside`` already uses (``OUTSIDE_X_RANGE`` by
``OUTSIDE_Y_RANGE``). A 2 cm grid covers that rectangle. A spot is kept for
training only when the scripted pick-and-handover can finish there. Two
groups of the remaining spots never enter the training demos:

* a scattered holdout, about 15% of the reachable spots, drawn with a fixed
  seed
* the whole far-Y corner of the outside band

The lists live in ``demo_sets/wider_spawn_holdouts.json``. With the wider
flag off, demo poses ignore this file and use gym-aloha's ``sample_box_pose``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mimic_arm.cube_pose import (
    CUBE_QUAT,
    CUBE_Z,
    DEFAULT_X_RANGE,
    DEFAULT_Y_RANGE,
    OUTSIDE_X_RANGE,
    OUTSIDE_Y_RANGE,
    in_default_box,
)

ROOT = Path(__file__).resolve().parents[1]
HOLDOUTS_PATH = ROOT / "demo_sets" / "wider_spawn_holdouts.json"
PLOT_PATH = ROOT / "demo_sets" / "wider_spawn_spots.png"

# 2 cm is half the 4 cm cube, so neighboring spots are distinct grasps.
GRID_STEP_M = 0.02
HOLDOUT_FRACTION = 0.15
HOLDOUT_SEED = 14
# Half the grid step. A leaked spot sits on this grid, so anything closer
# than 5 mm is that spot and not a neighbor 2 cm away.
LEAK_TOLERANCE_M = 0.005

# Far-Y corner of the outside band: the 5 cm square past both the high-Y
# edge (y = 0.6) and the high-X edge (x = 0.2). Edges of the training
# rectangle are inside that rectangle, so the corner starts strictly outside
# both. In-range misses on seeds 1024, 1038, 1047 and 1049 sat at
# x 0.130–0.161 and y 0.569–0.599, on the way into this square.
FAR_Y_CORNER_X = (0.2, 0.25)
FAR_Y_CORNER_Y = (0.6, 0.65)

# MuJoCo 3 leaves the stock solimp (0.25) about 7 cm short of the scripted
# waypoint, so the right gripper misses most of the original training box.
# The time constant stays 0.01. Only the impedance is raised, and only while
# the end-effector teacher is recording. Reachability is the joint replay in
# the stock joint-position env, which has no mocap weld.
STIFF_WELD_FROM = 'solref="0.01 1" solimp=".25 .25 0.001"'
STIFF_WELD_TO = 'solref="0.01 1" solimp="0.95 0.99 0.001"'


def in_far_y_corner(x: float, y: float) -> bool:
    """True for the high-X, high-Y square of the outside band."""
    return (
        FAR_Y_CORNER_X[0] < float(x) <= FAR_Y_CORNER_X[1]
        and FAR_Y_CORNER_Y[0] < float(y) <= FAR_Y_CORNER_Y[1]
    )


def candidate_spots(step: float = GRID_STEP_M) -> list[tuple[float, float]]:
    """Grid over the wider rectangle, sorted by x then y."""
    xs = np.round(np.arange(OUTSIDE_X_RANGE[0], OUTSIDE_X_RANGE[1] + 1e-9, step), 5)
    ys = np.round(np.arange(OUTSIDE_Y_RANGE[0], OUTSIDE_Y_RANGE[1] + 1e-9, step), 5)
    spots = [(float(x), float(y)) for x in xs for y in ys]
    return sorted(spots)


def _as_pairs(spots: list[tuple[float, float]] | np.ndarray) -> list[tuple[float, float]]:
    pairs = [(round(float(x), 5), round(float(y), 5)) for x, y in spots]
    return sorted(pairs)


def assign_holdouts(
    reachable: list[tuple[float, float]],
    *,
    fraction: float = HOLDOUT_FRACTION,
    seed: int = HOLDOUT_SEED,
) -> dict[str, list[tuple[float, float]]]:
    """Split reachable spots into training, scattered holdout, and corner.

    The scattered draw is ``round(fraction * reachable)`` spots, taken from
    the reachable spots that are not already in the far-Y corner. The corner
    eval list is the reachable grid points in that square. Unreachable corner
    points stay out of training and out of eval.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"holdout fraction must be between 0 and 1, got {fraction}.")
    reachable_pairs = _as_pairs(reachable)
    # The corner bounds stay fixed. Eval only lists the corner spots the
    # teacher can finish; the rest of that square is recorded separately.
    corner_grid = [spot for spot in candidate_spots() if in_far_y_corner(*spot)]
    corner_set = set(corner_grid)
    reachable_set = set(reachable_pairs)
    corner = [spot for spot in corner_grid if spot in reachable_set]
    corner_unreachable = [spot for spot in corner_grid if spot not in reachable_set]
    eligible = [spot for spot in reachable_pairs if spot not in corner_set]
    count = int(round(fraction * len(reachable_pairs)))
    if count > len(eligible):
        count = len(eligible)
    if count < 1 and eligible:
        raise ValueError("The scattered holdout would be empty.")
    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(eligible), size=count, replace=False)
    scattered = sorted(eligible[int(i)] for i in chosen)
    scattered_set = set(scattered)
    training = [
        spot
        for spot in reachable_pairs
        if spot not in scattered_set and spot not in corner_set
    ]
    return {
        "reachable": reachable_pairs,
        "scattered": scattered,
        "far_y_corner": corner,
        "far_y_corner_unreachable": corner_unreachable,
        "far_y_corner_grid": corner_grid,
        "training": training,
    }


def sample_box_pose_gym(seed: int) -> np.ndarray:
    """gym-aloha's ``sample_box_pose`` for one seed.

    One ``RandomState.uniform`` draws x, y, and z together. Two separate
    uniform calls would not match, so this stays in that one call. z is
    fixed at 0.05 by giving the uniform a zero-width range.
    """
    rng = np.random.RandomState(int(seed))
    ranges = np.array(
        [list(DEFAULT_X_RANGE), list(DEFAULT_Y_RANGE), [CUBE_Z, CUBE_Z]],
        dtype=float,
    )
    xyz = rng.uniform(ranges[:, 0], ranges[:, 1])
    return np.concatenate([xyz, CUBE_QUAT]).astype(np.float64)


def pose_from_xy(x: float, y: float) -> np.ndarray:
    return np.array([float(x), float(y), CUBE_Z, *CUBE_QUAT], dtype=np.float64)


@dataclass(frozen=True)
class SpawnPlan:
    reachable: tuple[tuple[float, float], ...]
    scattered: tuple[tuple[float, float], ...]
    far_y_corner: tuple[tuple[float, float], ...]
    training: tuple[tuple[float, float], ...]
    rejected: tuple[tuple[float, float], ...]
    raw: dict

    @property
    def candidate_count(self) -> int:
        return int(self.raw["candidate_count"])

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)


def _pairs_from_json(rows: list) -> tuple[tuple[float, float], ...]:
    return tuple((float(row[0]), float(row[1])) for row in rows)


def load_spawn_plan(path: Path | None = None) -> SpawnPlan:
    file = Path(path) if path is not None else HOLDOUTS_PATH
    raw = json.loads(file.read_text())
    rejected = _pairs_from_json(raw["rejected"])
    return SpawnPlan(
        reachable=_pairs_from_json(raw["reachable"]),
        scattered=_pairs_from_json(raw["scattered_holdout"]),
        far_y_corner=_pairs_from_json(raw["far_y_corner"]),
        training=_pairs_from_json(raw["training"]),
        rejected=rejected,
        raw=raw,
    )


def save_spawn_plan(payload: dict, path: Path | None = None) -> Path:
    file = Path(path) if path is not None else HOLDOUTS_PATH
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(payload, indent=2) + "\n")
    return file


def build_plan_payload(
    reachable: list[tuple[float, float]],
    *,
    joint_only_count: int | None = None,
) -> dict:
    """JSON document for the checked-in holdout file."""
    assigned = assign_holdouts(reachable)
    candidates = candidate_spots()
    reachable_set = set(assigned["reachable"])
    rejected = [spot for spot in candidates if spot not in reachable_set]
    outside_reachable = [spot for spot in assigned["reachable"] if not in_default_box(*spot)]
    outside_rejected = [spot for spot in rejected if not in_default_box(*spot)]
    in_box_candidates = [spot for spot in candidates if in_default_box(*spot)]
    in_box_reachable = [spot for spot in assigned["reachable"] if in_default_box(*spot)]
    return {
        "description": (
            "Scripted-demo cube spots for the wider spawn. Training demos use "
            "`training` only. `scattered_holdout` and `far_y_corner` are never "
            "training spots. Regenerated by assign_holdouts on `reachable` "
            f"with seed {HOLDOUT_SEED}."
        ),
        "seed": HOLDOUT_SEED,
        "holdout_fraction": HOLDOUT_FRACTION,
        "grid_step_m": GRID_STEP_M,
        "wider_x": list(OUTSIDE_X_RANGE),
        "wider_y": list(OUTSIDE_Y_RANGE),
        "default_x": list(DEFAULT_X_RANGE),
        "default_y": list(DEFAULT_Y_RANGE),
        "far_y_corner_bounds": {
            "x_exclusive_low": FAR_Y_CORNER_X[0],
            "x_inclusive_high": FAR_Y_CORNER_X[1],
            "y_exclusive_low": FAR_Y_CORNER_Y[0],
            "y_inclusive_high": FAR_Y_CORNER_Y[1],
            "note": (
                "High-X, high-Y 5 cm square of the outside band. "
                "In-range misses (seeds 1024, 1038, 1047, 1049) were at "
                "x 0.130-0.161, y 0.569-0.599."
            ),
        },
        "reachability": {
            "method": (
                "The scripted PickAndTransferPolicy is recorded in the "
                "end-effector sim with the stock solref (0.01) and a higher "
                "weld impedance (solimp 0.95 0.99). A spot is reachable when "
                "replaying those joint positions in the stock joint-position "
                "env reaches reward 4. That env is the eval sim and its weld "
                "is untouched. The stock solimp 0.25 leaves the right gripper "
                "about 7 cm short under MuJoCo 3, which rejects the right half "
                "of the original training box."
            ),
            "weld": STIFF_WELD_TO,
            "success_reward": 4,
        },
        "candidate_count": len(candidates),
        "reachable_count": len(assigned["reachable"]),
        "rejected_count": len(rejected),
        "joint_only_count": joint_only_count,
        "in_box_candidate_count": len(in_box_candidates),
        "in_box_reachable_count": len(in_box_reachable),
        "outside_band_reachable_count": len(outside_reachable),
        "outside_band_rejected_count": len(outside_rejected),
        "scattered_count": len(assigned["scattered"]),
        "far_y_corner_count": len(assigned["far_y_corner"]),
        "far_y_corner_unreachable_count": len(assigned["far_y_corner_unreachable"]),
        "training_count": len(assigned["training"]),
        "reachable": [list(spot) for spot in assigned["reachable"]],
        "rejected": [list(spot) for spot in rejected],
        "scattered_holdout": [list(spot) for spot in assigned["scattered"]],
        "far_y_corner": [list(spot) for spot in assigned["far_y_corner"]],
        "far_y_corner_unreachable": [list(spot) for spot in assigned["far_y_corner_unreachable"]],
        "training": [list(spot) for spot in assigned["training"]],
    }


def demo_cube_pose(episode_index: int, *, wider_spawn: bool, seed: int) -> np.ndarray:
    """Cube pose for one recorded episode.

    Flag off: gym-aloha's box and ``sample_box_pose(seed + episode_index)``.
    Flag on: the next saved training spot, in file order. The holdout file
    is not read when the flag is off.
    """
    if episode_index < 0:
        raise ValueError(f"episode_index must be >= 0, got {episode_index}.")
    if not wider_spawn:
        return sample_box_pose_gym(int(seed) + int(episode_index))
    plan = load_spawn_plan()
    if not plan.training:
        raise RuntimeError(f"No training spots in {HOLDOUTS_PATH}.")
    if episode_index >= len(plan.training):
        raise ValueError(
            f"Episode {episode_index} is past the {len(plan.training)} training spots. "
            "The wider recorder does not repeat or borrow a held-out spot."
        )
    x, y = plan.training[episode_index]
    return pose_from_xy(x, y)


def wider_training_poses(episode_count: int | None = None) -> list[np.ndarray]:
    """Poses the wider recorder would write, in order."""
    plan = load_spawn_plan()
    count = len(plan.training) if episode_count is None else int(episode_count)
    return [demo_cube_pose(i, wider_spawn=True, seed=0) for i in range(count)]


def holdout_spots(which: str) -> tuple[tuple[float, float], ...]:
    """``scattered`` or ``far-y`` list from the checked-in file."""
    plan = load_spawn_plan()
    if which in ("scattered", "holdout-scattered"):
        return plan.scattered
    if which in ("far-y", "far_y", "holdout-far-y"):
        return plan.far_y_corner
    raise ValueError(f"Unknown holdout list {which!r}.")


def leak_pairs(
    training: list[tuple[float, float]] | np.ndarray,
    holdouts: list[tuple[float, float]] | np.ndarray,
    tolerance_m: float = LEAK_TOLERANCE_M,
) -> list[tuple[tuple[float, float], tuple[float, float], float]]:
    """Training spots that land within ``tolerance_m`` of a holdout."""
    if tolerance_m <= 0:
        raise ValueError("tolerance must be positive.")
    leaked: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    hold = [(float(x), float(y)) for x, y in holdouts]
    for train_x, train_y in training:
        point = (float(train_x), float(train_y))
        for held_x, held_y in hold:
            distance = float(np.hypot(point[0] - held_x, point[1] - held_y))
            if distance <= tolerance_m:
                leaked.append((point, (held_x, held_y), distance))
    return leaked


def plot_spawn_plan(plan: SpawnPlan, path: Path) -> Path:
    """Training spots, scattered holdout, and the far-Y corner."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 6.2))

    def scatter(spots: tuple[tuple[float, float], ...], **kwargs) -> None:
        if not spots:
            return
        xs = [spot[0] for spot in spots]
        ys = [spot[1] for spot in spots]
        axis.scatter(xs, ys, **kwargs)

    scatter(plan.rejected, s=18, c="#d0d0d0", marker="x", linewidths=0.6, label=f"rejected ({len(plan.rejected)})", zorder=1)
    scatter(plan.training, s=28, c="#1f77b4", marker="o", label=f"training ({len(plan.training)})", zorder=2)
    scatter(plan.scattered, s=46, c="#ff7f0e", marker="^", label=f"scattered holdout ({len(plan.scattered)})", zorder=3)
    scatter(
        plan.far_y_corner,
        s=42,
        c="#d62728",
        marker="s",
        label=f"far-Y corner ({len(plan.far_y_corner)})",
        zorder=4,
    )

    def rectangle(bounds_x, bounds_y, **kwargs) -> None:
        width = bounds_x[1] - bounds_x[0]
        height = bounds_y[1] - bounds_y[0]
        axis.add_patch(plt.Rectangle((bounds_x[0], bounds_y[0]), width, height, fill=False, **kwargs))

    rectangle(OUTSIDE_X_RANGE, OUTSIDE_Y_RANGE, edgecolor="#666666", linewidth=1.0, label="wider box")
    rectangle(DEFAULT_X_RANGE, DEFAULT_Y_RANGE, edgecolor="black", linewidth=1.2, label="training box")
    rectangle(
        FAR_Y_CORNER_X,
        FAR_Y_CORNER_Y,
        edgecolor="#d62728",
        linewidth=1.0,
        linestyle="--",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("cube x (m)")
    axis.set_ylabel("cube y (m)")
    axis.set_title("Wider scripted spawn: training, holdouts, far-Y corner")
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    axis.set_xlim(OUTSIDE_X_RANGE[0] - 0.02, OUTSIDE_X_RANGE[1] + 0.04)
    axis.set_ylim(OUTSIDE_Y_RANGE[0] - 0.02, OUTSIDE_Y_RANGE[1] + 0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path
