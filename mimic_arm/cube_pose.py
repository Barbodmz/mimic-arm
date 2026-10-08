"""Where the red cube starts in AlohaTransferCube-v0.

gym-aloha's ``sample_box_pose`` draws x from 0.0 to 0.2 and y from 0.4 to
0.6, and pins z at 0.05. That is the distribution the demonstrations and a
normal evaluation use. ``outside`` draws from a 5 cm frame around that
rectangle, so the cube is off the training rectangle but still on the table
and in the top camera. The same seed always returns the same pose.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# Inclusive edges, matching gym_aloha.utils.sample_box_pose.
DEFAULT_X_RANGE = (0.0, 0.2)
DEFAULT_Y_RANGE = (0.4, 0.6)
CUBE_Z = 0.05
CUBE_QUAT = (1.0, 0.0, 0.0, 0.0)

# 5 cm past each edge of the training rectangle. Points inside the training
# rectangle are rejected, so samples land in the frame around it.
# Written as decimals so 0.4 - 0.05 does not become 0.35000000000000003.
OUTSIDE_MARGIN_M = 0.05
OUTSIDE_X_RANGE = (-0.05, 0.25)
OUTSIDE_Y_RANGE = (0.35, 0.65)


def in_default_box(x: float, y: float) -> bool:
    """True when (x, y) is inside the rectangle gym-aloha samples from."""
    return DEFAULT_X_RANGE[0] <= x <= DEFAULT_X_RANGE[1] and DEFAULT_Y_RANGE[0] <= y <= DEFAULT_Y_RANGE[1]


def _as_seed(seed) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, np.ndarray):
        if seed.size != 1:
            raise ValueError(f"Expected one seed, got shape {seed.shape}.")
        seed = seed.reshape(-1)[0]
    return int(seed)


def sample_transfer_cube_pose(
    seed,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    *,
    exclude_default: bool = False,
) -> np.ndarray:
    """Pose ``(x, y, z, qw, qx, qy, qz)`` for one episode.

    ``seed`` drives ``numpy.random.RandomState``, same as gym-aloha, so two
    checkpoints that share a seed share a cube. z stays 0.05, which is what
    ``sample_box_pose`` writes before the cube settles onto the table.
    """
    lo_x, hi_x = float(x_range[0]), float(x_range[1])
    lo_y, hi_y = float(y_range[0]), float(y_range[1])
    if not lo_x < hi_x:
        raise ValueError(f"cube x range must be low < high, got {x_range}.")
    if not lo_y < hi_y:
        raise ValueError(f"cube y range must be low < high, got {y_range}.")

    rng = np.random.RandomState(_as_seed(seed))
    for _ in range(10_000):
        x = float(rng.uniform(lo_x, hi_x))
        y = float(rng.uniform(lo_y, hi_y))
        if exclude_default and in_default_box(x, y):
            continue
        return np.array([x, y, CUBE_Z, *CUBE_QUAT], dtype=np.float64)
    raise RuntimeError(
        "Could not place a cube outside the training rectangle inside "
        f"x {x_range} and y {y_range}."
    )


class ListedCubeSampler:
    """Drop-in ``sample_box_pose`` that walks a fixed list of spots.

    Episode seed ``s`` uses ``spots[s % len(spots)]``, so a 50-episode eval
    cycles a shorter holdout list and every run with the same seed sees the
    same cubes.
    """

    def __init__(self, spots: list[tuple[float, float]], *, source: str) -> None:
        if not spots:
            raise ValueError("The holdout spot list is empty.")
        self.spots = tuple((float(x), float(y)) for x, y in spots)
        xs = [spot[0] for spot in self.spots]
        ys = [spot[1] for spot in self.spots]
        self.x_range = (min(xs), max(xs))
        self.y_range = (min(ys), max(ys))
        self.exclude_default = False
        self.source = source
        self.spot_count = len(self.spots)
        self.positions: dict[int, tuple[float, float]] = {}

    def __call__(self, seed=None) -> np.ndarray:
        key = _as_seed(seed)
        if key is None:
            raise ValueError("A holdout cube sampler needs a seed.")
        x, y = self.spots[key % len(self.spots)]
        self.positions[key] = (x, y)
        return np.array([x, y, CUBE_Z, *CUBE_QUAT], dtype=np.float64)


class RecordingCubeSampler:
    """Drop-in replacement for ``gym_aloha.env.sample_box_pose``.

    Remembers ``seed -> (x, y)`` so the eval log can print the position the
    simulator actually received.
    """

    def __init__(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        *,
        exclude_default: bool,
    ) -> None:
        self.x_range = (float(x_range[0]), float(x_range[1]))
        self.y_range = (float(y_range[0]), float(y_range[1]))
        self.exclude_default = exclude_default
        self.positions: dict[int, tuple[float, float]] = {}

    def __call__(self, seed=None) -> np.ndarray:
        pose = sample_transfer_cube_pose(
            seed,
            self.x_range,
            self.y_range,
            exclude_default=self.exclude_default,
        )
        key = _as_seed(seed)
        if key is not None:
            self.positions[key] = (float(pose[0]), float(pose[1]))
        return pose


HOLDOUT_CUBE_RANGES = ("holdout-scattered", "holdout-far-y")


def resolve_cube_sampler(
    cube_range: str,
    cube_x: tuple[float, float] | None,
    cube_y: tuple[float, float] | None,
) -> RecordingCubeSampler | ListedCubeSampler | None:
    """Return a sampler, or None when evaluation should use gym-aloha's own.

    None is the default. The caller must not patch the environment in that
    case, so a normal eval stays on ``sample_box_pose``.
    """
    allowed = ("default", "outside", *HOLDOUT_CUBE_RANGES)
    if cube_range not in allowed:
        raise ValueError(f"cube_range must be one of {allowed}, got {cube_range!r}.")
    if cube_range in HOLDOUT_CUBE_RANGES:
        if cube_x is not None or cube_y is not None:
            raise ValueError(
                f"{cube_range} reads its spots from the holdout file. "
                "Leave --cube-x and --cube-y unset."
            )
        from mimic_arm.wider_spawn import HOLDOUTS_PATH, holdout_spots

        return ListedCubeSampler(list(holdout_spots(cube_range)), source=str(HOLDOUTS_PATH))
    if (cube_x is None) != (cube_y is None):
        raise ValueError("Pass both cube x and cube y ranges, or neither.")
    if cube_range == "default" and cube_x is None:
        return None

    exclude_default = cube_range == "outside"
    if cube_x is None:
        x_range, y_range = OUTSIDE_X_RANGE, OUTSIDE_Y_RANGE
    else:
        x_range = (float(cube_x[0]), float(cube_x[1]))
        y_range = (float(cube_y[0]), float(cube_y[1]))
    return RecordingCubeSampler(x_range, y_range, exclude_default=exclude_default)


def cube_sampling_record(
    sampler: RecordingCubeSampler | ListedCubeSampler, cube_range: str
) -> dict:
    """JSON-ready description of the sampler that was installed."""
    record = {
        "mode": cube_range,
        "x": [sampler.x_range[0], sampler.x_range[1]],
        "y": [sampler.y_range[0], sampler.y_range[1]],
        "exclude_default": sampler.exclude_default,
        "default_x": [DEFAULT_X_RANGE[0], DEFAULT_X_RANGE[1]],
        "default_y": [DEFAULT_Y_RANGE[0], DEFAULT_Y_RANGE[1]],
        "z": CUBE_Z,
    }
    spot_count = getattr(sampler, "spot_count", None)
    if spot_count is not None:
        record["spot_count"] = int(spot_count)
        record["source"] = getattr(sampler, "source", None)
    return record


def install_cube_sampler(sampler: Callable):
    """Patch the function AlohaEnv.reset calls. Use as a context manager."""
    from contextlib import contextmanager

    @contextmanager
    def _patch():
        import gym_aloha.env as aloha_env

        original = aloha_env.sample_box_pose
        aloha_env.sample_box_pose = sampler
        try:
            yield
        finally:
            aloha_env.sample_box_pose = original

    return _patch()
