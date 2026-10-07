"""Motion-style similarity for human transfer-cube demonstrations.

Issue #12 compares the 20 most alike demos with 20 random ones. "Alike"
means the arms move in the same way: speed, pauses, and when the grippers
close. It does not mean the arms trace the same path through space.

Each demo is stretched onto a fixed number of samples so a longer recording
does not look different just because it has more frames. Duration and the
absolute time until the gripper closes are kept as their own terms, so
hesitation still counts, without letting length dominate the distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 14-d ALOHA layout shared with the dataset's observation.state / action.
LEFT_GRIPPER = 6
RIGHT_GRIPPER = 13
ARM_INDEX = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=int)

FPS = 50
COMMON_LENGTH = 100

# Normalized gripper: 0 is closed, 1 is open (gym-aloha). Demos start closed
# at the reset pose, then open on the way to the cube. The close that matters
# is the first one after that opening.
GRIPPER_OPEN = 0.75
GRIPPER_CLOSED = 0.70
GRIPPER_REOPEN = 0.85

# Arm-joint speed (rad/s, Euclidean over the 12 arm joints) below this is a pause.
PAUSE_SPEED_RAD_S = 0.20

# Block weights. Speed, pause, and gripper shape are each one RMS over the
# stretched curve, so a 100-sample curve does not outweigh a single timing
# number just by being long. Duration is one number on purpose.
WEIGHT_SPEED = 1.0
WEIGHT_PAUSE = 0.80
WEIGHT_RIGHT_GRIPPER = 1.0
WEIGHT_LEFT_GRIPPER = 0.60
WEIGHT_DURATION_S = 0.35
WEIGHT_RIGHT_CLOSE_S = 0.50
WEIGHT_LEFT_CLOSE_S = 0.25
WEIGHT_PAUSE_FRACTION = 0.35


@dataclass(frozen=True)
class MotionStyle:
    """How one demo moves, with the geometric path left out.

    ``speed`` and ``pause`` are stretched to ``COMMON_LENGTH`` samples.
    The gripper curves are stretched the same way. The ``*_s`` fields are
    absolute seconds, so two demos with the same shape but different
    hesitation do not collapse to distance zero.
    """

    speed: np.ndarray
    pause: np.ndarray
    right_gripper: np.ndarray
    left_gripper: np.ndarray
    duration_s: float
    right_close_s: float
    left_close_s: float
    pause_fraction: float
    right_close_fraction: float
    left_close_fraction: float


def gripper_close_onset(gripper: np.ndarray) -> int | None:
    """Frame where the gripper first closes after having opened.

    Returns None when the gripper never opens, or never closes again.
    The onset is the start of the close, not the later minimum.
    """
    signal = np.asarray(gripper, dtype=float)
    if signal.ndim != 1 or signal.size < 2:
        raise ValueError("gripper signal must be a 1-d series with at least 2 frames.")
    opened = np.flatnonzero(signal > GRIPPER_OPEN)
    if opened.size == 0:
        return None
    start = int(opened[0])
    closed = np.flatnonzero(signal[start:] < GRIPPER_CLOSED)
    if closed.size == 0:
        return None
    return start + int(closed[0])


def _close_fraction(gripper: np.ndarray) -> float:
    onset = gripper_close_onset(gripper)
    if onset is None:
        return 1.0
    return float(onset) / float(len(gripper) - 1)


def _resample(values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != (values.size,):
        raise ValueError("only 1-d series can be stretched.")
    if values.size < 2:
        raise ValueError("need at least 2 samples to stretch a series.")
    if values.size == n:
        return values.copy()
    source = np.linspace(0.0, 1.0, values.size)
    grid = np.linspace(0.0, 1.0, n)
    return np.interp(grid, source, values)


def _arm_speed(state: np.ndarray, fps: float) -> np.ndarray:
    arm = np.asarray(state, dtype=float)[:, ARM_INDEX]
    delta = np.diff(arm, axis=0)
    speed = np.linalg.norm(delta, axis=1) * float(fps)
    return np.concatenate([speed[:1], speed])


def motion_style_features(
    state: np.ndarray,
    action: np.ndarray | None = None,
    fps: float = FPS,
) -> MotionStyle:
    """Motion style of one episode.

    ``state`` and ``action`` are ``(T, 14)`` joint streams. Speed and pauses
    come from how the arm joints in ``state`` actually change. Gripper-close
    timing comes from ``action`` when it is present (the command the human
    sent) and from ``state`` otherwise. Absolute joint positions are not
    stored: adding a constant to the arm, or flipping the direction of a
    joint, does not change this result when the speed and the grippers match.
    """
    state = np.asarray(state, dtype=float)
    if state.ndim != 2 or state.shape[1] != 14 or state.shape[0] < 2:
        raise ValueError(f"state must have shape (T, 14) with T >= 2, got {state.shape}.")
    if action is None:
        command = state
    else:
        command = np.asarray(action, dtype=float)
        if command.shape != state.shape:
            raise ValueError(
                f"action shape {command.shape} does not match state shape {state.shape}."
            )
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")

    speed = _arm_speed(state, fps)
    pause = (speed < PAUSE_SPEED_RAD_S).astype(float)
    duration_s = float(state.shape[0] - 1) / float(fps)
    right_fraction = _close_fraction(command[:, RIGHT_GRIPPER])
    left_fraction = _close_fraction(command[:, LEFT_GRIPPER])
    return MotionStyle(
        speed=_resample(speed, COMMON_LENGTH),
        pause=_resample(pause, COMMON_LENGTH),
        right_gripper=_resample(state[:, RIGHT_GRIPPER], COMMON_LENGTH),
        left_gripper=_resample(state[:, LEFT_GRIPPER], COMMON_LENGTH),
        duration_s=duration_s,
        right_close_s=right_fraction * duration_s,
        left_close_s=left_fraction * duration_s,
        pause_fraction=float(pause.mean()),
        right_close_fraction=right_fraction,
        left_close_fraction=left_fraction,
    )


def _rms(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    return float(np.sqrt(np.mean(delta * delta)))


def motion_distance(left: MotionStyle, right: MotionStyle) -> float:
    """Weighted distance. Path geometry is not a term."""
    return (
        WEIGHT_SPEED * _rms(left.speed, right.speed)
        + WEIGHT_PAUSE * _rms(left.pause, right.pause)
        + WEIGHT_RIGHT_GRIPPER * _rms(left.right_gripper, right.right_gripper)
        + WEIGHT_LEFT_GRIPPER * _rms(left.left_gripper, right.left_gripper)
        + WEIGHT_DURATION_S * abs(left.duration_s - right.duration_s)
        + WEIGHT_RIGHT_CLOSE_S * abs(left.right_close_s - right.right_close_s)
        + WEIGHT_LEFT_CLOSE_S * abs(left.left_close_s - right.left_close_s)
        + WEIGHT_PAUSE_FRACTION * abs(left.pause_fraction - right.pause_fraction)
    )


def pairwise_distances(styles: list[MotionStyle]) -> np.ndarray:
    n = len(styles)
    distances = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            distances[i, j] = distances[j, i] = motion_distance(styles[i], styles[j])
    return distances


def most_alike_indices(distances: np.ndarray, k: int) -> tuple[int, tuple[int, ...]]:
    """Medoid and the ``k`` episodes closest to it, medoid included.

    The medoid is the episode with the smallest total distance to the
    others. Ties break toward the smaller index. This is the tight cluster
    around the typical motion, not a geometric neighborhood in the workspace.
    """
    distances = np.asarray(distances, dtype=float)
    n = int(distances.shape[0])
    if distances.shape != (n, n):
        raise ValueError("distances must be a square matrix.")
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and {n}, got {k}.")
    totals = distances.sum(axis=1)
    medoid = int(np.argmin(totals))
    order = np.argsort(distances[medoid], kind="mergesort")
    chosen = tuple(int(i) for i in order[:k])
    return medoid, chosen


def random_episode_indices(n: int, k: int, seed: int) -> tuple[int, ...]:
    """``k`` distinct episode indices from ``range(n)``, sorted.

    ``random.Random(seed)`` draws the sample, so the same seed on Python 3.12
    returns the same episodes. The result is sorted so the file is stable;
    the seed fixes which episodes, not the printed order.
    """
    import random

    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and {n}, got {k}.")
    chosen = random.Random(int(seed)).sample(range(int(n)), int(k))
    return tuple(sorted(int(i) for i in chosen))
