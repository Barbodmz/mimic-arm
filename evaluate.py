#!/usr/bin/env python
"""Roll out a trained ACT policy in the AlohaTransferCube simulator.

Loads a checkpoint from a local folder or a Hugging Face Hub model id, runs
several episodes, prints the success rate and average reward, and writes a
few episode videos.

A success is the gym-aloha definition: the left gripper is holding the red
cube off the table (reward reaches 4, and the episode ends). Reward at each
step is 0, 1, 2, 3, or 4 depending on how far the hand-off has gotten. The
"average reward" printed here is the average of the *sum* of rewards over an
episode, which is what LeRobot reports as ``avg_sum_reward``.

Examples
--------
Evaluate a checkpoint this repo just trained, for the default 20 episodes::

    python evaluate.py \\
        --checkpoint outputs/train/act_aloha_transfer_cube/checkpoints/last/pretrained_model

Smoke test (a couple of episodes is enough to see that the loop works)::

    python evaluate.py \\
        --checkpoint outputs/train/smoke_act/checkpoints/last \\
        --episodes 2 --videos 1 --device cpu \\
        --output-dir outputs/eval/smoke_act

Evaluate the published 80k-step policy from the Hub::

    python evaluate.py --checkpoint lerobot/act_aloha_sim_transfer_cube_human --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mimic_arm.mujoco_gl import configure_mujoco_rendering

ENV_TASK = "AlohaTransferCube-v0"


def resolve_checkpoint(checkpoint: str) -> str:
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
        return checkpoint

    candidates: list[Path] = [
        path,
        path / "pretrained_model",
        path / "checkpoints" / "last" / "pretrained_model",
    ]
    checkpoint_root = path / "checkpoints"
    if checkpoint_root.is_dir():
        numbered = sorted(
            (child for child in checkpoint_root.iterdir() if child.is_dir() and child.name.isdigit()),
            key=lambda child: int(child.name),
        )
        if numbered:
            candidates.append(numbered[-1] / "pretrained_model")

    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return str(candidate)

    raise SystemExit(
        f"No policy config.json under {path}. Pass a pretrained_model directory, "
        "a training output directory, or a Hugging Face model id."
    )


def _json_ready(value):
    """Turn NumPy scalars into plain Python so ``json.dump`` accepts them."""
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Local checkpoint path, training output directory, or Hugging Face model id.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="How many simulated episodes to run. Default: 20.",
    )
    parser.add_argument(
        "--videos",
        type=int,
        default=2,
        help="How many of those episodes to save as mp4. Default: 2.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Parallel simulator instances. Default 1, which is the safe choice on CPU.",
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
        default=Path("outputs/eval/act_aloha_transfer_cube"),
        help="Where videos and eval_info.json are written.",
    )
    parser.add_argument("--seed", type=int, default=1000, help="First episode seed. LeRobot's default is 1000.")
    return parser


def main(argv: list[str] | None = None) -> None:
    configure_mujoco_rendering()
    args = build_parser().parse_args(argv)

    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1.")
    if args.videos < 0:
        raise SystemExit("--videos cannot be negative.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested, but this machine has no visible GPU. Using cpu.")
        device = "cpu"

    policy_path = resolve_checkpoint(args.checkpoint)
    videos_to_save = min(args.videos, args.episodes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.output_dir / "videos"

    print("Checkpoint:", policy_path)
    print("Task:", ENV_TASK)
    print(f"Episodes: {args.episodes}    videos saved: {videos_to_save}    device: {device}")

    # Imports stay below the renderer setup. Importing lerobot pulls in
    # gym-aloha, which imports MuJoCo.
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
    from lerobot.envs.configs import AlohaEnv
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy_all
    from lerobot.utils.random_utils import set_seed
    from lerobot.utils.utils import init_logging

    init_logging()
    set_seed(args.seed)

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = Path(policy_path)
    policy_cfg.device = device

    env_cfg = AlohaEnv(task=ENV_TASK)
    # One simulator at a time unless the student asks for more. Async envs
    # fork extra processes, which is painful on a small CPU machine.
    envs = make_env(env_cfg, n_envs=args.batch_size, use_async_envs=args.batch_size > 1)

    started = time.perf_counter()
    try:
        policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
        policy.eval()

        # The saved processor pipeline normalizes images and moves tensors
        # onto the device. Override the device so a checkpoint trained on a
        # GPU still evaluates on CPU (and the other way around).
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=str(policy_cfg.pretrained_path),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, policy_cfg)

        with torch.no_grad():
            info = eval_policy_all(
                envs=envs,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=args.episodes,
                max_episodes_rendered=videos_to_save,
                videos_dir=videos_dir if videos_to_save else None,
                start_seed=args.seed,
                max_parallel_tasks=1,
            )
    finally:
        close_envs(envs)

    elapsed = time.perf_counter() - started
    overall = info["overall"]
    success_rate = float(overall["pc_success"])
    avg_reward = float(overall["avg_sum_reward"])
    video_paths = [str(path) for path in overall.get("video_paths", [])]

    info_path = args.output_dir / "eval_info.json"
    info_path.write_text(json.dumps(info, indent=2, default=_json_ready))

    # The lines a student (and the smoke test) should be able to find.
    print()
    print(f"Success rate: {success_rate:.1f}%  ({args.episodes} episodes)")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Average max reward: {float(overall['avg_max_reward']):.3f}")
    print(f"Runtime: {elapsed:.1f}s")
    if video_paths:
        print("Videos:")
        for path in video_paths:
            print(f"  {path}")
    else:
        print("Videos: none (pass --videos 1 or more to save mp4s)")
    print(f"Wrote {info_path}")


if __name__ == "__main__":
    main()
