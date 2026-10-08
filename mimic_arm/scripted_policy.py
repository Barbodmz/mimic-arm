"""Open-loop pick-and-handover used to record scripted transfer-cube demos.

This is the trajectory from the ACT simulator's ``PickAndTransferPolicy``
(tonyzhaozh/act ``scripted_policy.py``), which gym-aloha's end-effector task
was built to run. The right gripper approaches the cube, lifts it to a meet
point, and the left gripper takes it. ``inject_noise`` stays off, so one cube
pose always produces the same action sequence.
"""

from __future__ import annotations

import numpy as np

MEET_XYZ = np.array([0.0, 0.5, 0.25], dtype=float)
EPISODE_LEN = 400


def _axis_angle(axis: np.ndarray, degrees: float) -> np.ndarray:
    """Unit quaternion ``(w, x, y, z)`` for a right-handed rotation."""
    half = np.deg2rad(degrees) / 2.0
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return np.array([np.cos(half), *(np.sin(half) * axis)], dtype=float)


def _qmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product. Same order as ``pyquaternion``'s ``q1 * q2``."""
    w1, x1, y1, z1 = (float(v) for v in left)
    w2, x2, y2, z2 = (float(v) for v in right)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def _interpolate(current: dict, nxt: dict, t: int) -> tuple[np.ndarray, np.ndarray, float]:
    span = nxt["t"] - current["t"]
    frac = 0.0 if span == 0 else (t - current["t"]) / span
    xyz = current["xyz"] + (nxt["xyz"] - current["xyz"]) * frac
    quat = current["quat"] + (nxt["quat"] - current["quat"]) * frac
    gripper = current["gripper"] + (nxt["gripper"] - current["gripper"]) * frac
    return xyz, quat, float(gripper)


class PickAndTransferPolicy:
    """Scripted transfer-cube policy. Call it once per simulator step."""

    def __init__(self) -> None:
        self.step_count = 0
        self._left: list[dict] = []
        self._right: list[dict] = []
        self._current_left: dict | None = None
        self._current_right: dict | None = None

    def _generate(self, observation: dict) -> None:
        init_right = np.asarray(observation["mocap_pose_right"], dtype=float)
        init_left = np.asarray(observation["mocap_pose_left"], dtype=float)
        box_xyz = np.asarray(observation["env_state"], dtype=float)[:3]
        pick_quat = _qmul(init_right[3:], _axis_angle(np.array([0.0, 1.0, 0.0]), -60.0))
        meet_left_quat = _axis_angle(np.array([1.0, 0.0, 0.0]), 90.0)
        meet = MEET_XYZ
        self._left = [
            {"t": 0, "xyz": init_left[:3].copy(), "quat": init_left[3:].copy(), "gripper": 0.0},
            {"t": 100, "xyz": meet + np.array([-0.1, 0.0, -0.02]), "quat": meet_left_quat.copy(), "gripper": 1.0},
            {"t": 260, "xyz": meet + np.array([0.02, 0.0, -0.02]), "quat": meet_left_quat.copy(), "gripper": 1.0},
            {"t": 310, "xyz": meet + np.array([0.02, 0.0, -0.02]), "quat": meet_left_quat.copy(), "gripper": 0.0},
            {"t": 360, "xyz": meet + np.array([-0.1, 0.0, -0.02]), "quat": np.array([1.0, 0.0, 0.0, 0.0]), "gripper": 0.0},
            {"t": 400, "xyz": meet + np.array([-0.1, 0.0, -0.02]), "quat": np.array([1.0, 0.0, 0.0, 0.0]), "gripper": 0.0},
        ]
        self._right = [
            {"t": 0, "xyz": init_right[:3].copy(), "quat": init_right[3:].copy(), "gripper": 0.0},
            {"t": 90, "xyz": box_xyz + np.array([0.0, 0.0, 0.08]), "quat": pick_quat.copy(), "gripper": 1.0},
            {"t": 130, "xyz": box_xyz + np.array([0.0, 0.0, -0.015]), "quat": pick_quat.copy(), "gripper": 1.0},
            {"t": 170, "xyz": box_xyz + np.array([0.0, 0.0, -0.015]), "quat": pick_quat.copy(), "gripper": 0.0},
            {"t": 200, "xyz": meet + np.array([0.05, 0.0, 0.0]), "quat": pick_quat.copy(), "gripper": 0.0},
            {"t": 220, "xyz": meet.copy(), "quat": pick_quat.copy(), "gripper": 0.0},
            {"t": 310, "xyz": meet.copy(), "quat": pick_quat.copy(), "gripper": 1.0},
            {"t": 360, "xyz": meet + np.array([0.1, 0.0, 0.0]), "quat": pick_quat.copy(), "gripper": 1.0},
            {"t": 400, "xyz": meet + np.array([0.1, 0.0, 0.0]), "quat": pick_quat.copy(), "gripper": 1.0},
        ]

    def __call__(self, observation: dict) -> np.ndarray:
        if self.step_count == 0:
            self._generate(observation)
        if self._left[0]["t"] == self.step_count:
            self._current_left = self._left.pop(0)
        if self._right[0]["t"] == self.step_count:
            self._current_right = self._right.pop(0)
        if self._current_left is None or self._current_right is None:
            raise RuntimeError("The scripted policy was stepped before its first waypoint.")
        left_xyz, left_quat, left_grip = _interpolate(self._current_left, self._left[0], self.step_count)
        right_xyz, right_quat, right_grip = _interpolate(self._current_right, self._right[0], self.step_count)
        self.step_count += 1
        return np.concatenate(
            [left_xyz, left_quat, [left_grip], right_xyz, right_quat, [right_grip]]
        )
