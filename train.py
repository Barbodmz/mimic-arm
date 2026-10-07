#!/usr/bin/env python
"""Train an ACT policy on the Aloha "transfer the cube" human demonstrations.

This is a thin wrapper around LeRobot's ``lerobot-train`` command. It fills in
the dataset, the ACT policy, and the matching simulation task. ACT's
architecture (ResNet-18, chunk of 100 actions, learning rate 1e-5, and the
rest) stays at LeRobot's defaults — the same settings used for the published
``lerobot/act_aloha_sim_transfer_cube_human`` checkpoint.

Checkpoints are written so a crash does not fill the disk or throw away the
run. One full recovery checkpoint is refreshed on a short interval and
replaced in place. Permanent weights-only snapshots (no optimizer state) are
kept every ``--weights-every`` steps and are never deleted. Rerunning the
same command against the same ``--output-dir`` resumes from the recovery
checkpoint.

Examples
--------
Full training run (100,000 steps, the LeRobot default)::

    python train.py --device cuda

60,000 steps on a 6 GB Windows laptop. One recovery checkpoint every 2,000
steps, permanent weights every 10,000. Rerun this exact command to resume::

    python train.py --steps 60000 --batch-size 8 --device cuda --num-workers 1 \\
        --recovery-every 2000 --weights-every 10000 --min-free-gb 1 \\
        --env_eval_freq=0 --output-dir outputs/train/act_aloha_60k

Short run on a laptop CPU::

    python train.py --steps 200 --batch-size 2 --device cpu --num-workers 0 \\
        --output-dir outputs/train/smoke_act

Any extra ``--flag=value`` is forwarded to ``lerobot-train``. ``--save_freq``
sets the recovery interval when you pass it (otherwise ``--recovery-every``
does). ``--env_eval_freq=0`` is the default here so training does not also
roll out the simulator.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The repo root has to be importable even if the student is not sitting in it.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Choose EGL vs OSMesa before LeRobot (and therefore MuJoCo) is imported.
# Training itself does not render, but a full run can roll the policy out in
# the simulator every ``env_eval_freq`` steps.
from mimic_arm.mujoco_gl import configure_mujoco_rendering
from mimic_arm.saving import (
    DEFAULT_MIN_FREE_GB,
    DEFAULT_RECOVERY_EVERY,
    DEFAULT_WEIGHTS_EVERY,
    CheckpointKeeper,
    dataset_repo_id,
    default_num_workers,
    find_resume_config,
    read_training_step,
)

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
        help=(
            "Where checkpoints are written. Rerunning the same command resumes "
            "from checkpoints/recovery when that folder exists."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "DataLoader worker processes. Default: 1 on Windows, 4 elsewhere. "
            "Use 0 on a small CPU machine. Each Windows worker re-imports PyTorch."
        ),
    )
    parser.add_argument(
        "--recovery-every",
        type=int,
        default=DEFAULT_RECOVERY_EVERY,
        help=(
            "Steps between full recovery checkpoints. Default 2000. Only one is "
            "kept; the previous is replaced. Ignored when you pass --save_freq."
        ),
    )
    parser.add_argument(
        "--weights-every",
        type=int,
        default=DEFAULT_WEIGHTS_EVERY,
        help=(
            "Steps between permanent weights-only saves. Default 10000. "
            "These are never deleted. The final step is always saved."
        ),
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MIN_FREE_GB,
        help=(
            "Skip a permanent weights save when free disk is below this many "
            "gigabytes. Default 1. The recovery save still runs when it fits."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue from checkpoints/recovery in --output-dir. "
            "This is assumed when a recovery checkpoint is already there. "
            "--steps is the new finish line, not a restart."
        ),
    )
    return parser


def _resume_config_path(output_dir: Path) -> Path:
    """Path to the train_config.json that a resumed run should load."""
    config = find_resume_config(output_dir)
    if config is None:
        raise SystemExit(
            f"Cannot resume: no train_config.json under {output_dir / 'checkpoints'}. "
            "Train once first, or pass --config_path=... yourself."
        )
    return config


def _run_lerobot(command: list[str], keeper: CheckpointKeeper) -> None:
    """Hand the built command to ``lerobot-train`` and hook checkpoint saves."""
    sys.argv = ["lerobot-train", *command]
    from lerobot.scripts.lerobot_train import main as train_main
    from mimic_arm.saving import install_checkpoint_hooks

    install_checkpoint_hooks(keeper)
    train_main()


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

    num_workers = args.num_workers if args.num_workers is not None else default_num_workers()
    repo_id = dataset_repo_id(extra, DATASET_REPO_ID)

    command = [
        "--policy.type=act",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--policy.device={device}",
        f"--output_dir={args.output_dir}",
        f"--num_workers={num_workers}",
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

    # Mid-training rollouts load the simulator on top of the training process.
    # That spike is a common way to exhaust a laptop page file. Evaluate
    # afterwards with evaluate.py unless a frequency was requested.
    if not _flag_given(extra, "--env_eval_freq"):
        command.append("--env_eval_freq=0")

    # One queued batch per worker instead of LeRobot's four. Ignored when
    # there are no worker processes (PyTorch rejects prefetch_factor then).
    if num_workers > 0 and not _flag_given(extra, "--prefetch_factor"):
        command.append("--prefetch_factor=2")

    if _flag_given(extra, "--save_freq"):
        pass
    else:
        command.append(f"--save_freq={args.recovery_every}")

    resume_config = None
    if args.output_dir.exists():
        resume_config = find_resume_config(args.output_dir)
    if args.resume or resume_config is not None:
        if not args.resume:
            print(
                f"Found a checkpoint in {args.output_dir}. "
                "Resuming automatically. Pass a new --output-dir to start over."
            )
        config_path = resume_config if resume_config is not None else _resume_config_path(args.output_dir)
        saved_step = read_training_step(config_path.parent.parent)
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

    print("Training ACT on", repo_id)
    print("Simulation task:", ENV_TASK)
    print("Device:", device)
    print("lerobot-train", " ".join(command))

    keeper = CheckpointKeeper(
        output_dir=args.output_dir,
        total_steps=args.steps,
        weights_every=args.weights_every,
        min_free_bytes=int(args.min_free_gb * 1024**3),
    )
    _run_lerobot(command, keeper)


if __name__ == "__main__":
    main()
