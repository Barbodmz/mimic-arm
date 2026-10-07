"""Approximate cube position for demos that do not store one.

``lerobot/aloha_sim_transfer_cube_human`` has no cube field. The right arm
picks the cube up, so the cube is about where the right fingertips are at
the moment the gripper closes on it. The first top-camera frame is only a
backup, and only when a frame is actually available.

Spots that do not land near the table are rejected. Nothing here fills a
missing spot with a guess. Issue #12 is dropped when the alike set and the
random set cannot be given a comparable spread of real spots.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mimic_arm.cube_pose import DEFAULT_X_RANGE, DEFAULT_Y_RANGE
from mimic_arm.motion_style import GRIPPER_REOPEN, RIGHT_GRIPPER, gripper_close_onset

# gym-aloha puppet gripper, meters of finger travel. 0 = closed, 1 = open.
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844
FINGER_CLOSED = 0.021
FINGER_OPEN = 0.057

# Inner face of the finger mesh, meters along the finger-link x axis.
# Measured from gym-aloha's vx300s_10_custom_finger_left.stl after the geom
# transform in vx300s_right.xml (pos 0.005 -0.052 0, euler 3.14 1.57 0).
# The cube sits between the two pads, so the y/z offsets cancel and only
# this forward offset remains. It is the mesh measurement, not a fit to the
# spawn box.
FINGER_PAD_OFFSET_M = 0.01286679

# Top camera in gym-aloha's transfer-cube scene. It sits above the table
# looking straight down. MuJoCo's camera matrix for that pose is identity:
# image +x is world +x, image +y (down the frame) is world -y.
TOP_CAMERA_XYZ = (0.0, 0.6, 0.8)
TOP_FOVY_DEG = 78.0
TOP_WIDTH = 640
TOP_HEIGHT = 480
CUBE_PLANE_Z = 0.05

# Fairness gates. Spans are compared with the 0.20 m spawn rectangle.
SPAWN_MARGIN_M = 0.05
MIN_SPAN_FRACTION = 0.50
MIN_INSIDE_FRACTION = 0.80
MIN_RANGE_RATIO = 0.70
MIN_STD_RATIO = 0.60
MAX_MEAN_GAP_M = 0.05
MIN_QUADRANTS = 3
TABLE_Z_MAX_M = 0.15
GRASP_Z_MAX_M = 0.20
GRASP_XY_MARGIN_M = 0.08

NEXT_STEP_IF_UNFAIR = (
    "Close issue #12 (not planned) and retrain scripted demonstrations "
    "with a wider cube spawn, then rerun the unseen-position eval."
)


@dataclass(frozen=True)
class CubeSpot:
    episode: int
    x: float
    y: float
    z: float
    frame: int | None
    source: str


@dataclass(frozen=True)
class FairnessReport:
    passed: bool
    reason: str
    method: str
    details: dict


def unnormalize_gripper(normalized: float) -> float:
    """Map a 0–1 gripper command onto the puppet finger joint, in meters."""
    unit = float(np.clip(normalized, 0.0, 1.0))
    travel = PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE
    return unit * travel + PUPPET_GRIPPER_POSITION_CLOSE


def _inertial() -> str:
    # MuJoCo rejects a moving body with no mass. The value is only there so
    # the kinematic tree compiles; cube spots use body positions, not dynamics.
    return '<inertial pos="0 0 0" mass="0.1" diaginertia="1e-3 1e-3 1e-3"/>'


def _arm_body(prefix: str, x: float, y: float, yaw: float) -> str:
    # Joint axes and link lengths are the ViperX 300s chain from gym-aloha
    # (vx300s_left.xml / vx300s_right.xml). Mesh geoms are omitted: forward
    # kinematics only needs the joints.
    mass = _inertial()
    return f"""
    <body name="{prefix}" pos="{x} {y} 0" euler="0 0 {yaw}">
      <body name="{prefix}/shoulder_link" pos="0 0 0.079">{mass}
        <joint name="{prefix}/waist" axis="0 0 1" range="-3.14158 3.14158" limited="true"/>
        <body name="{prefix}/upper_arm_link" pos="0 0 0.04805">{mass}
          <joint name="{prefix}/shoulder" axis="0 1 0" range="-1.85005 1.25664" limited="true"/>
          <body name="{prefix}/upper_forearm_link" pos="0.05955 0 0.3">{mass}
            <joint name="{prefix}/elbow" axis="0 1 0" range="-1.76278 1.6057" limited="true"/>
            <body name="{prefix}/lower_forearm_link" pos="0.2 0 0">{mass}
              <joint name="{prefix}/forearm_roll" axis="1 0 0" range="-3.14158 3.14158" limited="true"/>
              <body name="{prefix}/wrist_link" pos="0.1 0 0">{mass}
                <joint name="{prefix}/wrist_angle" axis="0 1 0" range="-1.8675 2.23402" limited="true"/>
                <body name="{prefix}/gripper_link" pos="0.069744 0 0">{mass}
                  <joint name="{prefix}/wrist_rotate" axis="1 0 0" range="-3.14158 3.14158" limited="true"/>
                  <body name="{prefix}/left_finger_link" pos="0.0687 0 0">{mass}
                    <joint name="{prefix}/left_finger" type="slide" axis="0 1 0" range="0.021 0.057" limited="true"/>
                  </body>
                  <body name="{prefix}/right_finger_link" pos="0.0687 0 0">{mass}
                    <joint name="{prefix}/right_finger" type="slide" axis="0 1 0" range="-0.057 -0.021" limited="true"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    """


def kinematic_model_xml() -> str:
    """Mesh-free bimanual ViperX, same joint order as the simulator."""
    left = _arm_body("left", -0.469, 0.5, 0.0)
    right = _arm_body("right", 0.469, 0.5, 3.1416)
    return f"<mujoco><compiler angle=\"radian\"/><worldbody>{left}{right}</worldbody></mujoco>"


def _import_mujoco():
    """Import MuJoCo for forward kinematics, without starting a renderer.

    ``MUJOCO_GL`` is read at import time. Training sets it to ``osmesa`` on a
    CPU machine. Kinematics does not draw, so a fresh import uses ``disable``
    and then puts the previous value back for anything that renders later.
    """
    import os
    import sys

    loaded = sys.modules.get("mujoco")
    if loaded is not None:
        return loaded
    previous = os.environ.get("MUJOCO_GL")
    os.environ["MUJOCO_GL"] = "disable"
    try:
        import mujoco
    finally:
        if previous is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous
    return mujoco


class RightGripperKinematics:
    """Fingertip position of the right arm from a 14-d joint vector."""

    def __init__(self) -> None:
        mujoco = _import_mujoco()

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(kinematic_model_xml())
        self.data = mujoco.MjData(self.model)
        self._left_finger = self._body("right/left_finger_link")
        self._right_finger = self._body("right/right_finger_link")
        self._joint = {
            name: self._qposadr(name)
            for name in (
                "left/waist",
                "left/shoulder",
                "left/elbow",
                "left/forearm_roll",
                "left/wrist_angle",
                "left/wrist_rotate",
                "left/left_finger",
                "left/right_finger",
                "right/waist",
                "right/shoulder",
                "right/elbow",
                "right/forearm_roll",
                "right/wrist_angle",
                "right/wrist_rotate",
                "right/left_finger",
                "right/right_finger",
            )
        }

    def _body(self, name: str) -> int:
        index = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_BODY, name)
        if index < 0:
            raise RuntimeError(f"MuJoCo model is missing body {name}.")
        return int(index)

    def _qposadr(self, name: str) -> int:
        index = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, name)
        if index < 0:
            raise RuntimeError(f"MuJoCo model is missing joint {name}.")
        return int(self.model.jnt_qposadr[index])

    def cube_xyz(self, state: np.ndarray) -> np.ndarray:
        """World xyz of the point between the right fingertips."""
        joints = np.asarray(state, dtype=float)
        if joints.shape != (14,):
            raise ValueError(f"state must have shape (14,), got {joints.shape}.")
        qpos = np.zeros(self.model.nq, dtype=float)
        arm = {
            "left/waist": joints[0],
            "left/shoulder": joints[1],
            "left/elbow": joints[2],
            "left/forearm_roll": joints[3],
            "left/wrist_angle": joints[4],
            "left/wrist_rotate": joints[5],
            "right/waist": joints[7],
            "right/shoulder": joints[8],
            "right/elbow": joints[9],
            "right/forearm_roll": joints[10],
            "right/wrist_angle": joints[11],
            "right/wrist_rotate": joints[12],
        }
        for name, value in arm.items():
            qpos[self._joint[name]] = value
        left_finger = float(np.clip(unnormalize_gripper(joints[6]), FINGER_CLOSED, FINGER_OPEN))
        right_finger = float(np.clip(unnormalize_gripper(joints[13]), FINGER_CLOSED, FINGER_OPEN))
        qpos[self._joint["left/left_finger"]] = left_finger
        qpos[self._joint["left/right_finger"]] = -left_finger
        qpos[self._joint["right/left_finger"]] = right_finger
        qpos[self._joint["right/right_finger"]] = -right_finger
        self.data.qpos[:] = qpos
        self._mujoco.mj_forward(self.model, self.data)
        midpoint = 0.5 * (
            self.data.xpos[self._left_finger] + self.data.xpos[self._right_finger]
        )
        # xmat is row-major; column 0 is the finger-link x axis in world.
        axis = self.data.xmat[self._left_finger].reshape(3, 3)[:, 0]
        return np.asarray(midpoint + FINGER_PAD_OFFSET_M * axis, dtype=float)


def grasp_is_on_table(xyz: np.ndarray) -> bool:
    """True when this fingertip point could be a cube sitting on the table."""
    x, y, z = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    if not (0.0 <= z <= GRASP_Z_MAX_M):
        return False
    return (
        DEFAULT_X_RANGE[0] - GRASP_XY_MARGIN_M <= x <= DEFAULT_X_RANGE[1] + GRASP_XY_MARGIN_M
        and DEFAULT_Y_RANGE[0] - GRASP_XY_MARGIN_M <= y <= DEFAULT_Y_RANGE[1] + GRASP_XY_MARGIN_M
    )


def _close_window_end(gripper: np.ndarray, onset: int) -> int:
    reopen = np.flatnonzero(gripper[onset + 1 :] > GRIPPER_REOPEN)
    if reopen.size == 0:
        return len(gripper)
    return onset + 1 + int(reopen[0])


def cube_spot_from_gripper_close(
    state: np.ndarray,
    episode: int,
    kinematics: RightGripperKinematics | None = None,
) -> CubeSpot | None:
    """Cube spot from the right gripper's first close onto the table.

    Among frames in that close (until the gripper opens again), the frame
    with the lowest fingertip is the one resting on the cube. Returns None
    when the gripper never closes or the lowest point is not on the table.
    """
    state = np.asarray(state, dtype=float)
    onset = gripper_close_onset(state[:, RIGHT_GRIPPER])
    if onset is None:
        return None
    end = _close_window_end(state[:, RIGHT_GRIPPER], onset)
    kin = kinematics or RightGripperKinematics()
    best_xyz: np.ndarray | None = None
    best_frame: int | None = None
    for frame in range(onset, end):
        xyz = kin.cube_xyz(state[frame])
        if best_xyz is None or float(xyz[2]) < float(best_xyz[2]):
            best_xyz = xyz
            best_frame = frame
    if best_xyz is None or best_frame is None or not grasp_is_on_table(best_xyz):
        return None
    return CubeSpot(
        episode=int(episode),
        x=float(best_xyz[0]),
        y=float(best_xyz[1]),
        z=float(best_xyz[2]),
        frame=int(best_frame),
        source="gripper_close",
    )


def _focal_length_px() -> float:
    half = np.deg2rad(TOP_FOVY_DEG) / 2.0
    return (TOP_HEIGHT / 2.0) / float(np.tan(half))


def project_table_point(x: float, y: float, z: float = CUBE_PLANE_Z) -> tuple[float, float]:
    """Pixel (column, row) of a table point in the top camera."""
    depth = TOP_CAMERA_XYZ[2] - float(z)
    if depth <= 0:
        raise ValueError(f"point z={z} is not below the top camera.")
    focal = _focal_length_px()
    column = TOP_WIDTH / 2.0 + focal * (float(x) - TOP_CAMERA_XYZ[0]) / depth
    row = TOP_HEIGHT / 2.0 - focal * (float(y) - TOP_CAMERA_XYZ[1]) / depth
    return float(column), float(row)


def unproject_table_pixel(column: float, row: float, z: float = CUBE_PLANE_Z) -> tuple[float, float]:
    """World xy on the horizontal plane ``z`` for one top-camera pixel."""
    depth = TOP_CAMERA_XYZ[2] - float(z)
    focal = _focal_length_px()
    x = TOP_CAMERA_XYZ[0] + (float(column) - TOP_WIDTH / 2.0) * depth / focal
    y = TOP_CAMERA_XYZ[1] - (float(row) - TOP_HEIGHT / 2.0) * depth / focal
    return float(x), float(y)


def red_blob_centroid(image: np.ndarray) -> tuple[float, float] | None:
    """Column, row of a compact red blob. None when the frame has no cube-like blob."""
    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"image must be HxWx3, got {frame.shape}.")
    red = frame[:, :, 0].astype(np.int16)
    green = frame[:, :, 1].astype(np.int16)
    blue = frame[:, :, 2].astype(np.int16)
    mask = (red > 150) & (green < 80) & (blue < 80) & (red > green + 40) & (red > blue + 40)
    count = int(mask.sum())
    if count < 30 or count > 0.05 * mask.size:
        return None
    rows, cols = np.nonzero(mask)
    if float(cols.std()) > 40 or float(rows.std()) > 40:
        return None
    return float(cols.mean()), float(rows.mean())


def cube_spot_from_top_frame(image: np.ndarray, episode: int) -> CubeSpot | None:
    """Backup spot from the red cube in one top-camera frame.

    Returns None when no compact red blob is visible. A missing frame must
    not be replaced with a coordinate.
    """
    centroid = red_blob_centroid(image)
    if centroid is None:
        return None
    x, y = unproject_table_pixel(centroid[0], centroid[1], CUBE_PLANE_Z)
    xyz = np.array([x, y, CUBE_PLANE_Z], dtype=float)
    if not grasp_is_on_table(xyz):
        return None
    return CubeSpot(
        episode=int(episode),
        x=x,
        y=y,
        z=CUBE_PLANE_Z,
        frame=0,
        source="top_camera",
    )


def recover_cube_spot(
    state: np.ndarray,
    episode: int,
    first_frame: np.ndarray | None = None,
    kinematics: RightGripperKinematics | None = None,
) -> CubeSpot | None:
    """Gripper-close pose first. The camera frame is used only if that fails."""
    spot = cube_spot_from_gripper_close(state, episode, kinematics)
    if spot is not None:
        return spot
    if first_frame is None:
        return None
    return cube_spot_from_top_frame(first_frame, episode)


def _finite_spot(spot: CubeSpot | None) -> bool:
    if spot is None:
        return False
    return bool(np.isfinite([spot.x, spot.y, spot.z]).all())


def _inside_spawn(spot: CubeSpot) -> bool:
    return (
        DEFAULT_X_RANGE[0] - SPAWN_MARGIN_M <= spot.x <= DEFAULT_X_RANGE[1] + SPAWN_MARGIN_M
        and DEFAULT_Y_RANGE[0] - SPAWN_MARGIN_M <= spot.y <= DEFAULT_Y_RANGE[1] + SPAWN_MARGIN_M
    )


def quadrant(x: float, y: float) -> tuple[int, int]:
    """Spawn-rectangle quadrant. The split is the center of the training box."""
    center_x = 0.5 * (DEFAULT_X_RANGE[0] + DEFAULT_X_RANGE[1])
    center_y = 0.5 * (DEFAULT_Y_RANGE[0] + DEFAULT_Y_RANGE[1])
    return (int(x >= center_x), int(y >= center_y))


def recovery_problem(spots: Sequence[CubeSpot | None]) -> str | None:
    """Why these spots cannot be used. None when the cloud looks like the spawn box.

    Fails closed: a missing spot, a non-finite coordinate, a collapsed cloud,
    points off the table, or a cloud that misses the spawn rectangle.
    """
    n = len(spots)
    if n == 0:
        return "no episodes to recover cube spots from"
    missing = [i for i, spot in enumerate(spots) if not _finite_spot(spot)]
    if missing:
        return f"missing cube spot for {len(missing)} of {n} episodes"
    ready = [spot for spot in spots if spot is not None]
    xs = np.array([spot.x for spot in ready], dtype=float)
    ys = np.array([spot.y for spot in ready], dtype=float)
    zs = np.array([spot.z for spot in ready], dtype=float)
    spawn_x = DEFAULT_X_RANGE[1] - DEFAULT_X_RANGE[0]
    spawn_y = DEFAULT_Y_RANGE[1] - DEFAULT_Y_RANGE[0]
    if float(np.ptp(xs)) < MIN_SPAN_FRACTION * spawn_x:
        return "recovered x spread is too small to be cube spots"
    if float(np.ptp(ys)) < MIN_SPAN_FRACTION * spawn_y:
        return "recovered y spread is too small to be cube spots"
    inside = sum(1 for spot in ready if _inside_spawn(spot))
    if inside < MIN_INSIDE_FRACTION * n:
        return "recovered spots do not land on the cube spawn rectangle"
    if float(np.median(zs)) < 0.0 or float(np.median(zs)) > TABLE_Z_MAX_M:
        return "grasp height is not near the table, so these are not cube spots"
    return None


def _ratio(left: float, right: float) -> float:
    hi = max(float(left), float(right))
    if hi <= 1e-9:
        return 0.0
    return min(float(left), float(right)) / hi


def _spread_stats(spots: Sequence[CubeSpot]) -> dict:
    xs = np.array([spot.x for spot in spots], dtype=float)
    ys = np.array([spot.y for spot in spots], dtype=float)
    return {
        "x_span": float(np.ptp(xs)),
        "y_span": float(np.ptp(ys)),
        "x_std": float(xs.std()),
        "y_std": float(ys.std()),
        "x_mean": float(xs.mean()),
        "y_mean": float(ys.mean()),
        "quadrants": dict(Counter(quadrant(spot.x, spot.y) for spot in spots)),
    }


def spatial_mismatch(alike: Sequence[CubeSpot], random: Sequence[CubeSpot]) -> str | None:
    """None when the two clouds cover a comparable spread. Otherwise the reason."""
    if len(alike) == 0 or len(random) == 0:
        return "one of the sets has no cube spots"
    if len(alike) != len(random):
        return f"set sizes differ ({len(alike)} vs {len(random)})"
    a = _spread_stats(alike)
    b = _spread_stats(random)
    if _ratio(a["x_span"], b["x_span"]) < MIN_RANGE_RATIO:
        return "x ranges are not comparable"
    if _ratio(a["y_span"], b["y_span"]) < MIN_RANGE_RATIO:
        return "y ranges are not comparable"
    if _ratio(a["x_std"], b["x_std"]) < MIN_STD_RATIO:
        return "x spreads are not comparable"
    if _ratio(a["y_std"], b["y_std"]) < MIN_STD_RATIO:
        return "y spreads are not comparable"
    if abs(a["x_mean"] - b["x_mean"]) > MAX_MEAN_GAP_M:
        return "mean cube x differs by more than the allowed gap"
    if abs(a["y_mean"] - b["y_mean"]) > MAX_MEAN_GAP_M:
        return "mean cube y differs by more than the allowed gap"
    if len(a["quadrants"]) < MIN_QUADRANTS or len(b["quadrants"]) < MIN_QUADRANTS:
        return "one set misses too many quadrants of the spawn rectangle"
    return None


def coverage_matched_indices(
    distance_to_medoid: np.ndarray,
    spots: Sequence[CubeSpot | None],
    random_indices: Sequence[int],
    k: int,
) -> tuple[int, ...] | None:
    """Most typical episodes in each quadrant, with the random set's counts.

    Returns None when a quadrant the random set uses does not have enough
    recovered episodes. Does not copy the random indices: inside a quadrant
    the episodes closest to the medoid are kept.
    """
    if any(not _finite_spot(spots[i]) for i in random_indices):
        return None
    target = Counter(quadrant(spots[i].x, spots[i].y) for i in random_indices)  # type: ignore[union-attr]
    chosen: list[int] = []
    order = np.argsort(np.asarray(distance_to_medoid, dtype=float), kind="mergesort")
    for quad, count in sorted(target.items()):
        candidates = [
            int(i)
            for i in order
            if _finite_spot(spots[int(i)])
            and quadrant(spots[int(i)].x, spots[int(i)].y) == quad  # type: ignore[union-attr]
            and int(i) not in chosen
        ]
        if len(candidates) < count:
            return None
        chosen.extend(candidates[:count])
    if len(chosen) != k:
        return None
    return tuple(sorted(chosen))


def _mean_distance(distance_to_medoid: np.ndarray, indices: Sequence[int]) -> float:
    return float(np.mean(np.asarray(distance_to_medoid, dtype=float)[list(indices)]))


def fairness_problem(
    alike: Sequence[int],
    random_indices: Sequence[int],
    spots: Sequence[CubeSpot | None],
    distance_to_medoid: np.ndarray,
) -> str | None:
    """None when the two sets are a fair consistency-vs-variety comparison."""
    if set(alike) == set(random_indices):
        return "the alike set and the random set are the same episodes"
    if any(not _finite_spot(spots[i]) for i in list(alike) + list(random_indices)):
        return "missing cube spot in one of the sets"
    spatial = spatial_mismatch(
        [spots[i] for i in alike],  # type: ignore[misc]
        [spots[i] for i in random_indices],  # type: ignore[misc]
    )
    if spatial is not None:
        return spatial
    if _mean_distance(distance_to_medoid, alike) >= _mean_distance(distance_to_medoid, random_indices):
        return "the alike set is not tighter in motion style than the random set"
    return None


def _details(
    alike: Sequence[int],
    random_indices: Sequence[int],
    spots: Sequence[CubeSpot | None],
    distance_to_medoid: np.ndarray,
) -> dict:
    def pack(indices: Sequence[int]) -> dict | None:
        picked = [spots[i] for i in indices]
        if any(spot is None for spot in picked):
            return None
        stats = _spread_stats(picked)  # type: ignore[arg-type]
        stats["quadrants"] = {
            f"{'high' if a else 'low'}_x_{'high' if b else 'low'}_y": count
            for (a, b), count in stats["quadrants"].items()
        }
        stats["mean_distance_to_medoid"] = _mean_distance(distance_to_medoid, indices)
        return stats

    return {
        "alike": pack(alike),
        "random": pack(random_indices),
        "overlap": sorted(set(alike) & set(random_indices)),
    }


def select_demo_sets(
    distances: np.ndarray,
    spots: Sequence[CubeSpot | None],
    *,
    k: int = 20,
    seed: int = 12,
    random_indices: Sequence[int] | None = None,
) -> tuple[int | None, tuple[int, ...], tuple[int, ...], FairnessReport]:
    """Pick the alike set and the random set, then apply the fairness gate.

    The alike set starts as the ``k`` episodes closest to the medoid. When
    that neighborhood does not cover the random set's cube spots, a second
    try keeps the random set's quadrant counts but still prefers episodes
    close to the medoid. If neither try is fair, the report says so and no
    coordinates are invented.
    """
    from mimic_arm.motion_style import most_alike_indices, random_episode_indices

    distances = np.asarray(distances, dtype=float)
    n = int(distances.shape[0])
    if len(spots) != n:
        raise ValueError(f"expected {n} cube spots, got {len(spots)}.")
    medoid, alike = most_alike_indices(distances, k)
    if random_indices is None:
        random_set = random_episode_indices(n, k, seed)
    else:
        random_set = tuple(int(i) for i in random_indices)
        if len(random_set) != k or len(set(random_set)) != k:
            raise ValueError(f"random set must be {k} distinct episodes.")
        if any(i < 0 or i >= n for i in random_set):
            raise ValueError("random set contains an episode index outside the dataset.")
    distance_to_medoid = distances[medoid]
    problem = recovery_problem(spots)
    if problem is not None:
        report = FairnessReport(
            passed=False,
            reason=problem,
            method="not_selected",
            details={"next_step": NEXT_STEP_IF_UNFAIR, **_details(alike, random_set, spots, distance_to_medoid)},
        )
        return medoid, alike, random_set, report

    reason = fairness_problem(alike, random_set, spots, distance_to_medoid)
    if reason is None:
        report = FairnessReport(
            passed=True,
            reason="alike and random sets cover a comparable spread of cube spots",
            method="medoid_neighborhood",
            details=_details(alike, random_set, spots, distance_to_medoid),
        )
        return medoid, alike, random_set, report

    matched = coverage_matched_indices(distance_to_medoid, spots, random_set, k)
    if matched is None:
        report = FairnessReport(
            passed=False,
            reason=f"{reason}; coverage matching could not fill the random set's quadrants",
            method="not_selected",
            details={"next_step": NEXT_STEP_IF_UNFAIR, "unconstrained_reason": reason},
        )
        return medoid, alike, random_set, report
    matched_reason = fairness_problem(matched, random_set, spots, distance_to_medoid)
    if matched_reason is not None:
        report = FairnessReport(
            passed=False,
            reason=matched_reason,
            method="not_selected",
            details={
                "next_step": NEXT_STEP_IF_UNFAIR,
                "unconstrained_reason": reason,
                **_details(matched, random_set, spots, distance_to_medoid),
            },
        )
        return medoid, alike, random_set, report
    report = FairnessReport(
        passed=True,
        reason="quadrant-matched alike set covers the same cube-spot spread as the random set",
        method="coverage_matched",
        details={"unconstrained_reason": reason, **_details(matched, random_set, spots, distance_to_medoid)},
    )
    return medoid, matched, random_set, report
