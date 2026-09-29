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
        help="Continue training from output-dir/checkpoints/last.",
    )
    return parser


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
        config_path = (
            args.output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
        )
        if not config_path.is_file():
            raise SystemExit(
                f"Cannot resume: {config_path} does not exist. "
                "Train once first, or pass --config_path=... yourself."
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

    train_main()


if __name__ == "__main__":
    main()
