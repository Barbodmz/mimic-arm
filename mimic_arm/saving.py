"""Crash-safe checkpoints that stay small on disk.

LeRobot's own save writes a full checkpoint (weights plus optimizer state,
about 591 MB for this ACT policy) into a new numbered folder every
``save_freq`` steps and never deletes the old ones. On a laptop with a few
gigabytes free, that pile either fills the disk or, with the default
20,000-step interval, throws away a long run when the machine crashes.

This module keeps two kinds of save:

* ``checkpoints/recovery`` — one full resumable checkpoint. It is rewritten
  on LeRobot's save interval and swapped into place with a directory rename,
  so a crash in the middle of writing leaves the previous recovery readable.
* ``checkpoints/weights/<step>`` — the policy folder only (config, weights,
  normalization stats). These are never deleted. ``evaluate.py`` and
  ``compare.py`` open them directly.

Nothing here imports LeRobot, so the rotation and crash tests can run on CPU
without torch.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Callable

RECOVERY_NAME = "recovery"
STAGING_NAME = ".recovery_next"
PREV_NAME = "recovery_prev"
WEIGHTS_NAME = "weights"
LAST_NAME = "last"
# Written only after LeRobot's save returns. training_step.json is not a
# safe commit point: LeRobot writes it before the optimizer state.
COMMIT_NAME = ".mimic_arm_complete"

# Until a recovery folder exists to measure, assume a full ACT checkpoint is
# about 591 MB and ask for a little headroom so the write cannot fill the disk.
DEFAULT_RECOVERY_RESERVE_BYTES = 800 * 1024 * 1024
# fp32 weights for a 51.6M-parameter ACT policy are about 197 MiB.
DEFAULT_WEIGHTS_RESERVE_BYTES = 250 * 1024 * 1024
DEFAULT_RECOVERY_EVERY = 2000
DEFAULT_WEIGHTS_EVERY = 10_000
DEFAULT_MIN_FREE_GB = 1.0

Populate = Callable[[Path], None]
DiskUsage = Callable[[Path], object]


def default_num_workers(platform: str | None = None) -> int:
    """DataLoader workers. One on Windows, LeRobot's four everywhere else.

    Windows starts each worker with ``spawn``, and every worker re-imports
    PyTorch. That commit charge is what produced error 1455 (page file full)
    on a 6 GB laptop. One worker is the cheap way to cut it. Callers can
    still pass ``--num-workers 0``.
    """
    plat = sys.platform if platform is None else platform
    return 1 if plat == "win32" else 4


def step_dirname(step: int, total_steps: int) -> str:
    """Zero-padded step folder, matching LeRobot's ``get_step_identifier``."""
    digits = max(6, len(str(max(int(total_steps), int(step), 0))))
    return f"{step:0{digits}d}"


def should_save_weights(step: int, total_steps: int, weights_every: int) -> bool:
    """Permanent weights on the interval and always on the final step."""
    if step <= 0:
        return False
    if step == total_steps:
        return True
    if weights_every <= 0:
        return False
    return step % weights_every == 0


def format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.2f} GB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.0f} MB"
    return f"{num_bytes} bytes"


def dataset_repo_id(extra: list[str], default: str) -> str:
    """``--dataset.repo_id`` from forwarded CLI flags, or ``default``."""
    for index, token in enumerate(extra):
        if token == "--dataset.repo_id" and index + 1 < len(extra):
            return extra[index + 1]
        prefix = "--dataset.repo_id="
        if token.startswith(prefix):
            return token[len(prefix) :]
    return default


def has_checkpoint_files(path: Path) -> bool:
    """Train config plus the step file LeRobot writes under ``training_state``."""
    return (path / "pretrained_model" / "train_config.json").is_file() and (
        path / "training_state" / "training_step.json"
    ).is_file()


def is_complete_recovery(path: Path) -> bool:
    """A recovery tree that finished swapping into place.

    ``training_step.json`` is written before the optimizer file, so it is not
    enough. The commit marker is added only after the save function returns.
    """
    return has_checkpoint_files(path) and (path / COMMIT_NAME).is_file()


def read_training_step(checkpoint_dir: Path) -> int | None:
    step_file = checkpoint_dir / "training_state" / "training_step.json"
    if not step_file.is_file():
        return None
    data = json.loads(step_file.read_text())
    if "step" not in data:
        return None
    return int(data["step"])


def tree_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for file in path.rglob("*"):
        if file.is_file():
            total += file.stat().st_size
    return total


def fsync_tree(root: Path) -> None:
    """Flush a finished tree before renaming it into place."""
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            fd = os.open(Path(dirpath) / name, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        _fsync_dir(Path(dirpath))
    _fsync_dir(root.parent)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _discard_incomplete(path: Path) -> None:
    if path.exists() and not is_complete_recovery(path):
        shutil.rmtree(path)


def _discard_weight_temps(checkpoints: Path) -> None:
    weights = checkpoints / WEIGHTS_NAME
    if not weights.is_dir():
        return
    for child in weights.iterdir():
        if child.name.startswith(".tmp-"):
            shutil.rmtree(child)


def atomic_promote(source: Path, recovery: Path) -> None:
    """Replace ``recovery`` with the finished ``source`` tree.

    Rename is atomic on one filesystem. The previous recovery stays at
    ``recovery_prev`` until the new tree occupies ``recovery``, so a crash
    cannot leave zero good copies. ``source`` must not itself be
    ``recovery_prev`` (that swap is handled by :func:`install_best_recovery`).
    """
    parent = recovery.parent
    prev = parent / PREV_NAME
    if source.resolve() == prev.resolve():
        raise ValueError("atomic_promote source must not be recovery_prev")
    if recovery.exists():
        if prev.exists():
            shutil.rmtree(prev)
        recovery.rename(prev)
    source.rename(recovery)
    _fsync_dir(parent)
    if prev.exists():
        shutil.rmtree(prev)


def install_best_recovery(best: Path, recovery: Path) -> None:
    """Put ``best`` at ``recovery`` without deleting it first."""
    if best.resolve() == recovery.resolve():
        return
    parent = recovery.parent
    prev = parent / PREV_NAME
    if best.resolve() == prev.resolve():
        if not recovery.exists():
            prev.rename(recovery)
            _fsync_dir(parent)
            return
        # ``recovery`` is an older complete tree. Park it in staging only when
        # staging is not the tree we are keeping.
        bridge = parent / STAGING_NAME
        if bridge.exists():
            shutil.rmtree(bridge)
        recovery.rename(bridge)
        prev.rename(recovery)
        _fsync_dir(parent)
        shutil.rmtree(bridge)
        return
    atomic_promote(best, recovery)


def reconcile_recovery(checkpoints: Path) -> Path | None:
    """Make ``checkpoints/recovery`` the newest finished full checkpoint.

    Interrupted staging is deleted. A finished staging tree, or a recovery
    that was renamed aside mid-swap, is promoted. Returns the recovery
    directory when one is complete.
    """
    checkpoints.mkdir(parents=True, exist_ok=True)
    for name in (STAGING_NAME, PREV_NAME):
        _discard_incomplete(checkpoints / name)
    _discard_weight_temps(checkpoints)

    recovery = checkpoints / RECOVERY_NAME
    ranked: list[tuple[int, Path]] = []
    for path in (recovery, checkpoints / STAGING_NAME, checkpoints / PREV_NAME):
        if is_complete_recovery(path):
            ranked.append((read_training_step(path) or -1, path))
    if not ranked:
        return recovery if is_complete_recovery(recovery) else None

    # Equal steps: keep the tree already named recovery.
    ranked.sort(key=lambda item: (item[0], item[1].name == RECOVERY_NAME))
    best = ranked[-1][1]
    if best != recovery:
        install_best_recovery(best, recovery)
    for name in (STAGING_NAME, PREV_NAME):
        leftover = checkpoints / name
        if leftover.exists():
            shutil.rmtree(leftover)
    return recovery if is_complete_recovery(recovery) else None


def find_resume_config(output_dir: Path) -> Path | None:
    """``train_config.json`` to resume from, or None.

    Prefer the recovery checkpoint. Then LeRobot's ``last`` shortcut, then
    the newest numbered folder from a run that predates recovery saves.
    """
    output_dir = Path(output_dir)
    checkpoints = output_dir / "checkpoints"
    if not checkpoints.is_dir():
        return None

    recovery = reconcile_recovery(checkpoints)
    if recovery is not None:
        config = recovery / "pretrained_model" / "train_config.json"
        if config.is_file():
            return config

    last = checkpoints / LAST_NAME / "pretrained_model" / "train_config.json"
    if last.is_file():
        return last

    numbered = sorted(
        (
            child
            for child in checkpoints.iterdir()
            if child.is_dir() and child.name.isdigit() and has_checkpoint_files(child)
        ),
        key=lambda child: int(child.name),
    )
    if numbered:
        newest = numbered[-1]
        print(
            "checkpoints/recovery is missing, so resume will use the newest "
            f"numbered checkpoint ({newest.name})."
        )
        return newest / "pretrained_model" / "train_config.json"
    return None


class CheckpointKeeper:
    """Write recovery and weights folders for one training output directory."""

    def __init__(
        self,
        output_dir: Path,
        total_steps: int,
        weights_every: int = DEFAULT_WEIGHTS_EVERY,
        min_free_bytes: int = int(DEFAULT_MIN_FREE_GB * 1024**3),
        disk_usage: DiskUsage = shutil.disk_usage,
        recovery_reserve_bytes: int | None = None,
        weights_reserve_bytes: int = DEFAULT_WEIGHTS_RESERVE_BYTES,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.checkpoints = self.output_dir / "checkpoints"
        self.total_steps = int(total_steps)
        self.weights_every = int(weights_every)
        self.min_free_bytes = int(min_free_bytes)
        self.disk_usage = disk_usage
        self.recovery_reserve_bytes = recovery_reserve_bytes
        self.weights_reserve_bytes = int(weights_reserve_bytes)

    def free_bytes(self) -> int:
        probe = self.checkpoints if self.checkpoints.exists() else self.output_dir
        if not probe.exists():
            probe = self.output_dir.parent if self.output_dir.parent.exists() else Path.cwd()
        usage = self.disk_usage(probe)
        return int(getattr(usage, "free"))

    def recovery_bytes_needed(self) -> int:
        if self.recovery_reserve_bytes is not None:
            return int(self.recovery_reserve_bytes)
        recovery = self.checkpoints / RECOVERY_NAME
        if is_complete_recovery(recovery):
            return int(tree_size(recovery) * 1.1) + 1
        return DEFAULT_RECOVERY_RESERVE_BYTES

    def save_recovery(self, step: int, populate: Populate) -> bool:
        """Write one full checkpoint into ``recovery``, replacing the last one.

        ``populate`` must create a complete tree in the directory it is given.
        Returns False when the save is skipped or fails. A failure does not
        raise: training keeps the previous recovery.
        """
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        reconcile_recovery(self.checkpoints)
        staging = self.checkpoints / STAGING_NAME
        if staging.exists():
            shutil.rmtree(staging)

        free = self.free_bytes()
        need = self.recovery_bytes_needed()
        if free < need:
            print(
                "WARNING: Not enough free disk for a recovery checkpoint "
                f"({format_bytes(free)} free, about {format_bytes(need)} needed). "
                "Skipping this recovery save so training can continue. "
                "Free some space and rerun the same train.py command to resume "
                "from the last good recovery."
            )
            return False

        staging.mkdir(parents=True)
        try:
            populate(staging)
        except OSError as exc:
            print(
                f"WARNING: recovery save failed ({exc}). "
                "Keeping the previous recovery checkpoint."
            )
            if staging.exists():
                shutil.rmtree(staging)
            return False

        if not has_checkpoint_files(staging):
            print(
                "WARNING: recovery save did not finish (training_step.json is missing). "
                "Keeping the previous recovery checkpoint."
            )
            shutil.rmtree(staging)
            return False

        # LeRobot can leave training_step.json on disk before the optimizer
        # file is finished. The marker is the commit point, and it is written
        # only after populate returns.
        (staging / COMMIT_NAME).write_text("ok\n")
        fsync_tree(staging)
        atomic_promote(staging, self.checkpoints / RECOVERY_NAME)
        print(f"Recovery checkpoint updated at step {step}: {self.checkpoints / RECOVERY_NAME}")
        self._maybe_save_weights(step)
        return True

    def _maybe_save_weights(self, step: int) -> bool:
        if not should_save_weights(step, self.total_steps, self.weights_every):
            return False
        dirname = step_dirname(step, self.total_steps)
        dest = self.checkpoints / WEIGHTS_NAME / dirname
        pretrained = dest / "pretrained_model"
        if (pretrained / "config.json").is_file() and not (dest / "training_state").exists():
            return True

        free = self.free_bytes()
        if free < self.min_free_bytes or free < self.weights_reserve_bytes:
            threshold = max(self.min_free_bytes, self.weights_reserve_bytes)
            print(
                "WARNING: "
                f"{format_bytes(free)} free is below the disk guard "
                f"({format_bytes(threshold)}). Skipping the permanent weights save "
                f"at step {step}. The recovery checkpoint was still updated if it "
                "fit. Permanent weights are never deleted automatically."
            )
            return False

        src = self.checkpoints / RECOVERY_NAME / "pretrained_model"
        if not (src / "config.json").is_file():
            print("WARNING: recovery has no policy files. Skipping the weights save.")
            return False

        weights_root = self.checkpoints / WEIGHTS_NAME
        weights_root.mkdir(parents=True, exist_ok=True)
        tmp = weights_root / f".tmp-{dirname}"
        if tmp.exists():
            shutil.rmtree(tmp)
        if dest.exists():
            shutil.rmtree(dest)
        try:
            shutil.copytree(src, tmp / "pretrained_model")
            fsync_tree(tmp)
            tmp.rename(dest)
            _fsync_dir(weights_root)
        except OSError as exc:
            print(
                f"WARNING: permanent weights save at step {step} failed ({exc}). "
                "The recovery checkpoint is intact."
            )
            if tmp.exists():
                shutil.rmtree(tmp)
            return False
        print(
            f"Saved permanent weights at step {step} to {dest} "
            "(optimizer state not included)."
        )
        return True

    def refresh_last_symlink(self) -> None:
        """Point ``checkpoints/last`` at ``recovery``.

        Creating the symlink raises ``OSError`` on Windows without Developer
        Mode or an admin account (WinError 1314), and on Google Drive. The
        recovery folder is the real checkpoint, so that failure is logged and
        ignored.
        """
        recovery = self.checkpoints / RECOVERY_NAME
        if not is_complete_recovery(recovery):
            return
        last = self.checkpoints / LAST_NAME
        try:
            if last.is_symlink():
                last.unlink()
            elif last.exists():
                print(
                    "checkpoints/last exists and is not a shortcut, so it was left "
                    "in place. Resume uses checkpoints/recovery."
                )
                return
            last.symlink_to(RECOVERY_NAME, target_is_directory=True)
        except OSError as exc:
            print(
                "Could not create the checkpoints/last shortcut "
                f"({exc}). The recovery checkpoint was still saved. "
                "Rerunning the same train.py command resumes from checkpoints/recovery."
            )


def install_checkpoint_hooks(keeper: CheckpointKeeper) -> None:
    """Send LeRobot's checkpoint writes through ``keeper``.

    LeRobot still decides *when* to save (``save_freq``, plus the final step).
    The numbered directory it picked is not used: the full tree is written to
    staging and swapped into ``checkpoints/recovery``. ``checkpoints/last``
    points at that recovery folder when the filesystem allows a symlink.
    """
    import lerobot.common.train_utils as train_utils
    import lerobot.scripts.lerobot_train as train_script

    original_save = train_utils.save_checkpoint

    def save_checkpoint(checkpoint_dir, step, *args, **kwargs):
        def populate(dest: Path) -> None:
            original_save(dest, step, *args, **kwargs)

        keeper.save_recovery(int(step), populate)

    def update_last_checkpoint(checkpoint_dir):
        keeper.refresh_last_symlink()
        return keeper.checkpoints / RECOVERY_NAME

    train_utils.save_checkpoint = save_checkpoint
    train_utils.update_last_checkpoint = update_last_checkpoint
    train_script.save_checkpoint = save_checkpoint
    train_script.update_last_checkpoint = update_last_checkpoint
