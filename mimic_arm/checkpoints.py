"""Find a trained policy folder inside a LeRobot training run.

``evaluate.py`` and ``compare.py`` both accept a Hugging Face model id, a
``pretrained_model`` directory, one checkpoint folder, or a whole training
output directory. This module is the shared way to turn those into a folder
that contains ``config.json``.

Current training runs store:

* ``checkpoints/recovery`` — the newest full checkpoint (also has optimizer state)
* ``checkpoints/weights/<step>`` — permanent weights-only snapshots
* ``checkpoints/last`` — a shortcut to ``recovery``, when the disk allows a symlink

Older runs stored numbered folders such as ``checkpoints/000200`` directly.
"""

from __future__ import annotations

import json
from pathlib import Path


def resolve_checkpoint(checkpoint: str | Path) -> str:
    """Return a folder (or Hub id) that contains ``config.json``.

    Accepted inputs:

    * a Hub id such as ``lerobot/act_aloha_sim_transfer_cube_human``
    * a ``pretrained_model`` directory
    * a step directory such as ``checkpoints/weights/010000`` or ``checkpoints/000200``
    * ``checkpoints/recovery``
    * a training output directory (the latest saved policy is used)
    """
    path = Path(checkpoint)
    if not path.exists():
        # Not a local folder, so treat it as a Hub model id. LeRobot downloads
        # config.json and model.safetensors when the policy is built.
        return str(checkpoint)

    if _is_checkpoint_collection(path):
        saved = list_numbered_checkpoints(path)
        policy = _policy_dir(saved[-1])
        if policy is None:
            raise SystemExit(
                f"No policy config.json under {saved[-1]}. "
                "The newest checkpoint is missing its pretrained_model folder."
            )
        return str(policy)

    policy = _policy_dir(path)
    if policy is not None:
        return str(policy)

    raise SystemExit(
        f"No policy config.json under {path}. Pass a pretrained_model directory, "
        "a training output directory, a weights folder, or a Hugging Face model id."
    )


def list_numbered_checkpoints(train_dir: str | Path) -> list[Path]:
    """Return saved policies, oldest first.

    Includes permanent ``checkpoints/weights/<step>`` folders, numbered
    folders from older runs, and ``checkpoints/recovery`` when its step is
    not already in that list. ``checkpoints/last`` is a shortcut, so it is
    not included a second time. Pass either the training output directory or
    the ``checkpoints`` directory itself.
    """
    path = Path(train_dir)
    root = path / "checkpoints" if (path / "checkpoints").is_dir() else path
    if not root.is_dir():
        raise SystemExit(f"No checkpoints directory under {path}.")

    by_step: dict[int, Path] = {}
    for child in _numbered_checkpoint_dirs(root):
        if _policy_dir(child) is not None:
            by_step[int(child.name)] = child

    weights = root / "weights"
    if weights.is_dir():
        for child in _numbered_checkpoint_dirs(weights):
            if _policy_dir(child) is not None:
                by_step[int(child.name)] = child

    recovery = root / "recovery"
    if _policy_dir(recovery) is not None:
        step = _step_from_training_state(recovery)
        if step is None:
            step = checkpoint_step(recovery)
        if step is not None and step not in by_step:
            by_step[step] = recovery

    if not by_step:
        raise SystemExit(
            f"No checkpoints in {root}. "
            "Expected checkpoints/weights/<step>, checkpoints/recovery, "
            "or a numbered folder such as 000200."
        )
    return [by_step[step] for step in sorted(by_step)]


def checkpoint_step(checkpoint: str | Path) -> int | None:
    """Training step for a checkpoint path, or None if it cannot be read.

    A folder named ``000200`` or ``weights/010000`` is that step. Otherwise
    the step is read from ``training_state/training_step.json``, which is
    what a recovery checkpoint writes next to ``pretrained_model``.
    """
    path = Path(checkpoint)
    candidates = [path, *path.parents]
    for candidate in candidates[:6]:
        if candidate.name.isdigit():
            return int(candidate.name)
        step = _step_from_training_state(candidate)
        if step is not None:
            return step
    return None


def _policy_dir(path: Path) -> Path | None:
    if (path / "config.json").is_file():
        return path
    pretrained = path / "pretrained_model"
    if (pretrained / "config.json").is_file():
        return pretrained
    return None


def _is_checkpoint_collection(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "checkpoints").is_dir():
        return True
    if path.name != "checkpoints":
        return False
    if (path / "weights").is_dir() or (path / "recovery").is_dir():
        return True
    return any(child.is_dir() and child.name.isdigit() for child in path.iterdir())


def _step_from_training_state(path: Path) -> int | None:
    step_file = path / "training_state" / "training_step.json"
    if not step_file.is_file():
        return None
    data = json.loads(step_file.read_text())
    if "step" not in data:
        return None
    return int(data["step"])


def _numbered_checkpoint_dirs(checkpoint_root: Path) -> list[Path]:
    if not checkpoint_root.is_dir():
        return []
    numbered = [
        child
        for child in checkpoint_root.iterdir()
        if child.is_dir() and child.name.isdigit()
    ]
    numbered.sort(key=lambda child: int(child.name))
    return numbered
