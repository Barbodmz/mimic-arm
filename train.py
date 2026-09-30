#!/usr/bin/env python
"""Train an ACT policy on the Aloha "transfer the cube" human demonstrations.

This is a thin wrapper around LeRobot's ``lerobot-train`` command. It fills in
the dataset, the ACT policy, and the matching simulation task. ACT's
architecture (ResNet-18, chunk of 100 actions, learning rate 1e-5, and the
rest) stays at LeRobot's defaults — the same settings used for the published
``lerobot/act_aloha_sim_transfer_cube_human`` checkpoint.

Examples
--------
Full training run (100,000 steps, the LeRobot default)::

    python train.py --device cuda

Short run on a laptop CPU::

    python train.py --steps 200 --batch-size 2 --device cpu --num-workers 0 \\
        --output-dir outputs/train/smoke_act

Any extra ``--flag=value`` is forwarded to ``lerobot-train``. For example,
``--save_freq=2000`` writes a checkpoint every 2,000 steps, and
``--env_eval_freq=0`` skips rollouts during training.

Continue a run and stop at a later step. ``--steps`` is the new finish line.
The checkpoint already knows how far training got::

    python train.py --resume --steps 20000 \\
        --output-dir outputs/train/act_aloha_transfer_cube --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The repo root has to be importable even if the student is not sitting in it.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Choose EGL vs OSMesa before LeRobot (and therefore MuJoCo) is imported.
# Training itself does not render, but a full run rolls the policy out in the
# simulator every ``env_eval_freq`` steps.
from mimic_arm.mujoco_gl import configure_mujoco_rendering

# The human demonstrations: 50 episodes of a person transferring a red cube
# from the right arm to the left arm in the gym-aloha simulator.
DATASET_REPO_ID = "lerobot/aloha_sim_transfer_cube_human"
# Must match the dataset. AlohaEnv's own default task is AlohaInsertion-v0.
ENV_TASK = "AlohaTransferCube-v0"


def _flag_given(extra: list[str], name: str) -> bool:
    """True if the student already passed this LeRobot flag in ``extra``."""
    prefix = name + "="
    return any(token == name or token.startswith(prefix) for token in extra)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100_000,
        help="Optimizer steps. LeRobot's default is 100000. The published checkpoint is at 80000.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Samples per step. LeRobot's default is 8. Lower it if you run out of memory.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=None,
        help="cuda or cpu. Default: cuda when a GPU is visible, otherwise cpu.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/train/act_aloha_transfer_cube"),
        help="Where checkpoints are written. LeRobot refuses to overwrite an existing directory.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes. LeRobot's default is 4. Use 0 on a small CPU machine.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue training from output-dir/checkpoints/last. "
            "--steps is the new finish line, not a restart."
        ),
    )
    return parser


def _resume_config_path(output_dir: Path) -> Path:
    """Path to the train_config.json that ``--resume`` should load.

    LeRobot's ``checkpoints/last`` entry is a symlink. Google Drive (and some
    other network disks) cannot store that symlink, so fall back to the
    highest numbered checkpoint folder. The numbered folder holds the same
    ``train_config.json`` and ``training_state``.
    """
    checkpoints = output_dir / "checkpoints"
    last = checkpoints / "last" / "pretrained_model" / "train_config.json"
    if last.is_file():
        return last

    numbered: list[Path] = []
    if checkpoints.is_dir():
        numbered = sorted(
            (child for child in checkpoints.iterdir() if child.is_dir() and child.name.isdigit()),
            key=lambda child: int(child.name),
        )
    for child in reversed(numbered):
        candidate = child / "pretrained_model" / "train_config.json"
        if candidate.is_file():
            print(
                "checkpoints/last is missing, so resume will use the newest "
                f"numbered checkpoint ({child.name})."
            )
            return candidate

    raise SystemExit(
        f"Cannot resume: no train_config.json under {checkpoints}. "
        "Train once first, or pass --config_path=... yourself."
    )


def _read_training_step(config_path: Path) -> int | None:
    """Step stored next to this train_config.json, if the file is there."""
    step_file = config_path.parent.parent / "training_state" / "training_step.json"
    if not step_file.is_file():
        return None
    data = json.loads(step_file.read_text())
    if "step" not in data:
        return None
    return int(data["step"])


def _keep_training_if_last_symlink_fails() -> None:
    """Let a checkpoint save finish when the disk cannot create ``last``.

    LeRobot creates ``checkpoints/last`` with ``Path.symlink_to`` after the
    numbered folder is already written. That call raises ``OSError`` on
    Google Drive, and on Windows without permission to create a symlink
    (WinError 1314). The numbered folder is enough for ``train.py --resume``.
    This has to run on a fresh training start as well as ``--resume``: the
    crash happens on the first checkpoint save, before any resume.
    """
    import lerobot.common.train_utils as train_utils
    import lerobot.scripts.lerobot_train as train_script

    original = train_utils.update_last_checkpoint

    def update_last_checkpoint(checkpoint_dir):
        try:
            return original(checkpoint_dir)
        except OSError as exc:
            print(
                "Could not create the checkpoints/last shortcut "
                f"({exc}). The numbered checkpoint folder was still saved. "
                "python train.py --resume will use the newest numbered folder."
            )
            return checkpoint_dir

    train_utils.update_last_checkpoint = update_last_checkpoint
    train_script.update_last_checkpoint = update_last_checkpoint


def _default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def main(argv: list[str] | None = None) -> None:
    configure_mujoco_rendering()
    parser = build_parser()
    # Unknown flags (``--save_freq=2000``, ``--wandb.enable=true``, ...) are
    # forwarded to LeRobot instead of being rejected.
    args, extra = parser.parse_known_args(argv)

    device = args.device or _default_device()
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            print("CUDA was requested, but this machine has no visible GPU. Using cpu.")
            device = "cpu"

    command = [
        "--policy.type=act",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--policy.device={device}",
        f"--output_dir={args.output_dir}",
        f"--num_workers={args.num_workers}",
        "--save_checkpoint=true",
    ]
    # Defaults match the transfer-cube lesson. A flag the student already
    # passed (for example their own ``--dataset.repo_id``) is left alone.
    if not _flag_given(extra, "--dataset.repo_id"):
        command.append(f"--dataset.repo_id={DATASET_REPO_ID}")
    if not _flag_given(extra, "--env.type"):
        command.append("--env.type=aloha")
    if not _flag_given(extra, "--env.task"):
        command.append(f"--env.task={ENV_TASK}")

    # LeRobot's default is to upload the policy, and training errors out if
    # you have not given it a Hub repo id. Keep checkpoints local unless the
    # student opts in with ``--policy.push_to_hub=true --policy.repo_id=user/name``.
    if not _flag_given(extra, "--policy.push_to_hub"):
        command.append("--policy.push_to_hub=false")

    if args.resume:
        config_path = _resume_config_path(args.output_dir)
        saved_step = _read_training_step(config_path)
        if saved_step is not None and args.steps <= saved_step:
            raise SystemExit(
                f"This checkpoint is already at step {saved_step}, and --steps is {args.steps}. "
                "Training would stop immediately. Pass a larger --steps, which is the new "
                f"finish line (for example --steps {saved_step + 10000})."
            )
        if saved_step is not None:
            print(
                f"Resuming from step {saved_step}. "
                f"Training continues until step {args.steps} "
                f"({args.steps - saved_step} more steps)."
            )
        command.append("--resume=true")
        if not _flag_given(extra, "--config_path"):
            command.append(f"--config_path={config_path}")

    command.extend(extra)

    print("Training ACT on", DATASET_REPO_ID)
    print("Simulation task:", ENV_TASK)
    print("Device:", device)
    print("lerobot-train", " ".join(command))

    # ``lerobot-train`` reads its configuration from sys.argv. Point argv at
    # the command we just built, then call the same function the console
    # script calls.
    sys.argv = ["lerobot-train", *command]
    from lerobot.scripts.lerobot_train import main as train_main

    # Every save, including the first one on a fresh run, tries to create
    # checkpoints/last. Catch that failure here so training does not die
    # after the numbered checkpoint is already on disk.
    _keep_training_if_last_symlink_fails()
    train_main()


if __name__ == "__main__":
    main()
