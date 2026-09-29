"""Find a trained policy folder inside a LeRobot training run.

``evaluate.py`` and ``compare.py`` both accept a Hugging Face model id, a
``pretrained_model`` directory, one checkpoint folder, or a whole training
output directory. This module is the shared way to turn those into a folder
that contains ``config.json``.
"""

from __future__ import annotations

import json
from pathlib import Path


def resolve_checkpoint(checkpoint: str | Path) -> str:
    """Return a folder (or Hub id) that contains ``config.json``.

    Accepted inputs:

    * a Hub id such as ``lerobot/act_aloha_sim_transfer_cube_human``
    * a ``pretrained_model`` directory
    * a step directory such as ``checkpoints/000200``
    * a training output directory (the latest checkpoint is used)
    """
    path = Path(checkpoint)
    if not path.exists():
        # Not a local folder, so treat it as a Hub model id. LeRobot downloads
        # config.json and model.safetensors when the policy is built.
        return str(checkpoint)

    candidates: list[Path] = [
        path,
        path / "pretrained_model",
        path / "checkpoints" / "last" / "pretrained_model",
    ]
    checkpoint_root = path / "checkpoints"
    if checkpoint_root.is_dir():
        numbered = _numbered_checkpoint_dirs(checkpoint_root)
        if numbered:
            candidates.append(numbered[-1] / "pretrained_model")

    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return str(candidate)

    raise SystemExit(
        f"No policy config.json under {path}. Pass a pretrained_model directory, "
        "a training output directory, or a Hugging Face model id."
    )


def list_numbered_checkpoints(train_dir: str | Path) -> list[Path]:
    """Return step folders such as ``checkpoints/000200``, oldest first.

    ``checkpoints/last`` is a shortcut to one of those folders, so it is not
    included a second time. Pass either the training output directory or the
    ``checkpoints`` directory itself.
    """
    path = Path(train_dir)
    root = path / "checkpoints" if (path / "checkpoints").is_dir() else path
    if not root.is_dir():
        raise SystemExit(f"No checkpoints directory under {path}.")
    numbered = _numbered_checkpoint_dirs(root)
    if not numbered:
        raise SystemExit(
            f"No numbered checkpoints in {root}. "
            "A folder named last is a shortcut, not an extra checkpoint."
        )
    return numbered


def checkpoint_step(checkpoint: str | Path) -> int | None:
    """Training step for a checkpoint path, or None if it cannot be read.

    A folder named ``000200`` is step 200. Otherwise the step is read from
    ``training_state/training_step.json``, which is what LeRobot writes next
    to ``pretrained_model``.
    """
    path = Path(checkpoint)
    candidates = [path, *path.parents]
    for candidate in candidates[:6]:
        if candidate.name.isdigit():
            return int(candidate.name)
        step_file = candidate / "training_state" / "training_step.json"
        if step_file.is_file():
            data = json.loads(step_file.read_text())
            if "step" in data:
                return int(data["step"])
    return None


def _numbered_checkpoint_dirs(checkpoint_root: Path) -> list[Path]:
    numbered = [
        child
        for child in checkpoint_root.iterdir()
        if child.is_dir() and child.name.isdigit()
    ]
    numbered.sort(key=lambda child: int(child.name))
    return numbered
