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

``--save-failures`` keeps a video of every episode that did not succeed, plus
one success for comparison. LeRobot 0.6.1 can only render the first N
episodes (it decides before it knows the outcome), so this flag renders every
episode and then deletes the extra success videos.

Examples
--------
Evaluate a checkpoint this repo just trained, for the default 20 episodes::

    python evaluate.py \\
        --checkpoint outputs/train/act_aloha_transfer_cube/checkpoints/last/pretrained_model

Save every failed episode (and one success) from a 50-episode run::

    python evaluate.py \\
        --checkpoint outputs/train/act_aloha_transfer_cube/checkpoints/last/pretrained_model \\
        --episodes 50 --save-failures --device cuda

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
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mimic_arm.checkpoints import checkpoint_step, resolve_checkpoint
from mimic_arm.cube_pose import cube_sampling_record, install_cube_sampler, resolve_cube_sampler
from mimic_arm.mujoco_gl import configure_mujoco_rendering

ENV_TASK = "AlohaTransferCube-v0"

# gym-aloha's transfer-cube reward. The max over an episode says how far the
# hand-off got, even when the episode is not a success.
REWARD_STAGES = (
    (0, "nothing"),
    (1, "touched"),
    (2, "lifted"),
    (3, "both grippers"),
    (4, "success"),
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


def reward_stage(max_reward: float) -> int:
    """Integer stage closest to an episode's maximum reward."""
    return int(round(float(max_reward)))


def format_reward(value: float) -> str:
    """Print 1 instead of 1.000 when the reward is a whole number."""
    number = float(value)
    rounded = round(number)
    if abs(number - rounded) < 1e-6:
        return str(int(rounded))
    return f"{number:.3f}"


def per_episode_rows(info: dict, start_seed: int) -> list[dict]:
    """One row per episode: seed, success, max_reward, sum_reward.

    ``eval_policy`` in LeRobot 0.6.1 records those four fields, but
    ``eval_policy_all`` only keeps three lists on each task: ``successes``,
    ``max_rewards``, and ``sum_rewards``. Seeds are dropped there. Episode i
    still used seed ``start_seed + i``, because the evaluator hands out seeds
    in that order and only discards the unused tail of the last batch.
    """
    tasks = info.get("per_task") or []
    if not tasks:
        raise RuntimeError(
            "eval_policy_all did not return per-task results, so this script "
            "cannot build the per-episode table."
        )

    rows: list[dict] = []
    for task in tasks:
        metrics = task.get("metrics") or {}
        successes = list(metrics.get("successes") or [])
        max_rewards = list(metrics.get("max_rewards") or [])
        sum_rewards = list(metrics.get("sum_rewards") or [])
        seeds = list(metrics.get("seeds") or [])
        count = min(len(successes), len(max_rewards), len(sum_rewards))
        for index in range(count):
            if index < len(seeds) and seeds[index] is not None:
                seed = int(seeds[index])
            else:
                seed = start_seed + index
            rows.append(
                {
                    "episode": len(rows),
                    "seed": seed,
                    "success": bool(successes[index]),
                    "max_reward": float(max_rewards[index]),
                    "sum_reward": float(sum_rewards[index]),
                }
            )
    return rows


def stage_table(rows: list[dict]) -> list[dict]:
    """Count episodes by the furthest reward stage they reached."""
    counts = {stage: 0 for stage, _label in REWARD_STAGES}
    extra: dict[int, int] = {}
    for row in rows:
        stage = reward_stage(row["max_reward"])
        if stage in counts:
            counts[stage] += 1
        else:
            extra[stage] = extra.get(stage, 0) + 1
    table = [
        {"stage": stage, "label": label, "count": counts[stage]}
        for stage, label in REWARD_STAGES
    ]
    for stage in sorted(extra):
        table.append({"stage": stage, "label": "other", "count": extra[stage]})
    return table


def format_episode_line(row: dict) -> str:
    outcome = "success" if row["success"] else "fail"
    line = (
        f"episode {row['episode']:02d}  seed {row['seed']}  {outcome:7}  "
        f"max_reward {format_reward(row['max_reward'])}  "
        f"sum_reward {row['sum_reward']:.3f}"
    )
    if "cube_x" in row:
        line += f"  cube ({row['cube_x']:.3f}, {row['cube_y']:.3f})"
    return line


def format_stage_report(table: list[dict]) -> str:
    lines = ["Failures by max reward stage:"]
    for row in table:
        lines.append(f"  {row['stage']} {row['label']}: {row['count']}")
    return "\n".join(lines)


def failure_video_name(row: dict) -> str:
    """File name for one episode video, for example ``episode_07_fail_maxreward1.mp4``."""
    episode = f"{row['episode']:02d}"
    if row["success"]:
        return f"episode_{episode}_success.mp4"
    stage = reward_stage(row["max_reward"])
    return f"episode_{episode}_fail_maxreward{stage}.mp4"


def _rendered_video(videos_dir: Path, episode_index: int) -> Path | None:
    """Find LeRobot's ``eval_episode_<index>.mp4`` for one episode."""
    name = f"eval_episode_{episode_index}.mp4"
    matches = sorted(videos_dir.rglob(name))
    if not matches:
        return None
    return matches[0]


def organize_failure_videos(videos_dir: Path, rows: list[dict]) -> list[str]:
    """Rename rendered videos and keep every failure plus one success.

    LeRobot always writes ``eval_episode_0.mp4``, ``eval_episode_1.mp4``, ...
    in episode order. It has no switch for "only the failures", so the caller
    renders every episode and this function does the selection afterwards.
    """
    kept: list[str] = []
    saved_success = False
    for row in rows:
        source = _rendered_video(videos_dir, row["episode"])
        if source is None:
            print(f"No video file for episode {row['episode']:02d}.")
            continue
        if row["success"] and saved_success:
            source.unlink()
            continue
        if row["success"]:
            saved_success = True
        destination = videos_dir / failure_video_name(row)
        if destination.exists() and destination.resolve() != source.resolve():
            destination.unlink()
        if destination.resolve() != source.resolve():
            source.rename(destination)
        kept.append(str(destination))

    for child in list(videos_dir.iterdir()):
        if child.is_dir() and not any(child.iterdir()):
            child.rmdir()
    return kept


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
        help="How many of those episodes to save as mp4. Default: 2. Ignored with --save-failures.",
    )
    parser.add_argument(
        "--save-failures",
        action="store_true",
        help=(
            "Save a video of every failed episode and at most one success. "
            "Names look like episode_07_fail_maxreward1.mp4 and episode_03_success.mp4."
        ),
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
    add_eval_variant_args(parser)
    return parser


def add_eval_variant_args(parser: argparse.ArgumentParser) -> None:
    """Flags that change evaluation only. Both default off.

    ``compare.py`` adds the same flags so a table can use them. Leaving them
    off keeps the previous evaluation path.
    """
    parser.add_argument(
        "--temporal-ensemble",
        type=float,
        default=None,
        metavar="COEFF",
        help=(
            "ACT temporal-ensemble coefficient, for example 0.01 (the ACT paper). "
            "Off by default. Sets n_action_steps to 1, so the network runs every "
            "simulator step. Much slower than the normal action chunk."
        ),
    )
    parser.add_argument(
        "--cube-range",
        choices=("default", "outside", "holdout-scattered", "holdout-far-y"),
        default="default",
        help=(
            "Where the red cube starts. 'default' is gym-aloha's sample_box_pose "
            "(x 0.0-0.2, y 0.4-0.6). 'outside' is a 5 cm frame around that rectangle "
            "(the original unseen-position comparison). 'holdout-scattered' and "
            "'holdout-far-y' walk demo_sets/wider_spawn_holdouts.json. Seed s uses "
            "spot s mod the list length, so each mode is its own run."
        ),
    )
    parser.add_argument(
        "--cube-x",
        nargs=2,
        type=float,
        metavar=("LO", "HI"),
        help="Explicit cube x range in meters. Pass with --cube-y.",
    )
    parser.add_argument(
        "--cube-y",
        nargs=2,
        type=float,
        metavar=("LO", "HI"),
        help="Explicit cube y range in meters. Pass with --cube-x.",
    )


def _finite(value: float) -> bool:
    number = float(value)
    return number == number and number not in (float("inf"), float("-inf"))


def evaluate_checkpoint(
    checkpoint: str,
    *,
    episodes: int,
    videos: int,
    batch_size: int,
    device: str | None,
    output_dir: Path,
    seed: int,
    save_failures: bool,
    temporal_ensemble: float | None = None,
    cube_range: str = "default",
    cube_x: tuple[float, float] | None = None,
    cube_y: tuple[float, float] | None = None,
) -> dict:
    """Run ``episodes`` rollouts and write ``eval_info.json``. Returns a summary."""
    configure_mujoco_rendering()

    if episodes < 1:
        raise SystemExit("--episodes must be at least 1.")
    if videos < 0:
        raise SystemExit("--videos cannot be negative.")
    if batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")

    import torch

    chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if chosen_device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested, but this machine has no visible GPU. Using cpu.")
        chosen_device = "cpu"

    policy_path = resolve_checkpoint(checkpoint)
    # Selective rendering is not available: max_episodes_rendered always saves
    # the first N episodes, before anyone knows which ones failed. Render all
    # of them when the caller asked to keep the failures.
    if save_failures:
        videos_to_save = episodes
    else:
        videos_to_save = min(videos, episodes)

    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    if videos_to_save and videos_dir.exists():
        for old_video in videos_dir.rglob("*.mp4"):
            old_video.unlink()

    print("Checkpoint:", policy_path)
    step = checkpoint_step(policy_path)
    if step is not None:
        print("Step:", step)
    print("Task:", ENV_TASK)
    if save_failures:
        print(
            f"Episodes: {episodes}    videos: every failure, plus one success    "
            f"device: {chosen_device}"
        )
    else:
        print(f"Episodes: {episodes}    videos saved: {videos_to_save}    device: {chosen_device}")

    try:
        sampler = resolve_cube_sampler(cube_range, cube_x, cube_y)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    cube_record = cube_sampling_record(sampler, cube_range) if sampler is not None else None
    if temporal_ensemble is not None and not _finite(temporal_ensemble):
        raise SystemExit("--temporal-ensemble must be a finite number, for example 0.01.")
    if temporal_ensemble is not None:
        print(
            f"Temporal ensemble: coefficient {float(temporal_ensemble):g}, "
            "n_action_steps 1. The network runs every step. "
            "The ensembler resets at the start of each episode."
        )
    if cube_record is not None:
        line = (
            f"Cube positions: {cube_record['mode']}  "
            f"x {cube_record['x']}  y {cube_record['y']}"
        )
        if cube_record["exclude_default"]:
            line += "  (training rectangle excluded)"
        if cube_record.get("spot_count") is not None:
            line += f"  ({cube_record['spot_count']} listed spots, {cube_record['source']})"
        print(line)

    # Imports stay below the renderer setup. Importing lerobot pulls in
    # gym-aloha, which imports MuJoCo.
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
    from lerobot.envs.configs import AlohaEnv
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy_all
    from lerobot.utils.random_utils import set_seed
    from lerobot.utils.utils import init_logging

    init_logging()
    set_seed(seed)

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = Path(policy_path)
    policy_cfg.device = chosen_device
    if temporal_ensemble is not None:
        if not isinstance(policy_cfg, ACTConfig):
            raise SystemExit("--temporal-ensemble applies to ACT checkpoints only.")
        # ACTConfig requires n_action_steps == 1 whenever temporal_ensemble_coeff
        # is set. __post_init__ already ran on the saved config, so run it again
        # after these two fields change.
        policy_cfg.temporal_ensemble_coeff = float(temporal_ensemble)
        policy_cfg.n_action_steps = 1
        policy_cfg.__post_init__()

    env_cfg = AlohaEnv(task=ENV_TASK)
    # One simulator at a time unless the student asks for more. Async envs
    # fork extra processes, which is painful on a small CPU machine.
    # A custom cube sampler is installed in this process, so those envs stay
    # synchronous too. Otherwise the worker processes would keep gym-aloha's sampler.
    use_async_envs = batch_size > 1 and sampler is None
    if sampler is not None and batch_size > 1:
        print("Cube sampling runs in this process, so the simulators are not async.")

    # nullcontext leaves gym-aloha's sample_box_pose in place.
    cube_patch = nullcontext() if sampler is None else install_cube_sampler(sampler)
    with cube_patch:
        envs = make_env(env_cfg, n_envs=batch_size, use_async_envs=use_async_envs)

        started = time.perf_counter()
        try:
            policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
            policy.eval()
            if temporal_ensemble is not None:
                # lerobot_eval.rollout calls policy.reset() before every episode.
                # For ACT that clears ACTTemporalEnsembler. Check the object exists
                # and that reset actually empties it before the first episode.
                ensembler = getattr(policy, "temporal_ensembler", None)
                if ensembler is None:
                    raise SystemExit("Temporal ensembling did not attach to this policy.")
                policy.reset()
                if ensembler.ensembled_actions is not None:
                    raise SystemExit("The temporal ensembler did not reset.")

            # The saved processor pipeline normalizes images and moves tensors
            # onto the device. Override the device so a checkpoint trained on a
            # GPU still evaluates on CPU (and the other way around).
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=policy_cfg,
                pretrained_path=str(policy_cfg.pretrained_path),
                preprocessor_overrides={"device_processor": {"device": chosen_device}},
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
                    n_episodes=episodes,
                    max_episodes_rendered=videos_to_save,
                    videos_dir=videos_dir if videos_to_save else None,
                    start_seed=seed,
                    max_parallel_tasks=1,
                )
        finally:
            close_envs(envs)

    elapsed = time.perf_counter() - started
    overall = info["overall"]
    rows = per_episode_rows(info, seed)
    if len(rows) != episodes:
        raise RuntimeError(
            f"Expected {episodes} per-episode results, got {len(rows)}."
        )
    stages = stage_table(rows)
    if sampler is not None:
        for row in rows:
            position = sampler.positions.get(row["seed"])
            if position is None:
                raise RuntimeError(
                    f"No cube position was recorded for episode seed {row['seed']}."
                )
            row["cube_x"], row["cube_y"] = position

    if save_failures:
        video_paths = organize_failure_videos(videos_dir, rows)
    else:
        video_paths = [str(path) for path in overall.get("video_paths") or []]

    overall["video_paths"] = video_paths
    info["per_episode"] = rows
    info["failure_stage_counts"] = stages
    if temporal_ensemble is not None:
        info["temporal_ensemble_coeff"] = float(temporal_ensemble)
        info["n_action_steps"] = 1
    if cube_record is not None:
        info["cube_sampling"] = cube_record

    info_path = output_dir / "eval_info.json"
    info_path.write_text(json.dumps(info, indent=2, default=_json_ready))

    success_rate = float(overall["pc_success"])
    avg_reward = float(overall["avg_sum_reward"])
    avg_max_reward = float(overall["avg_max_reward"])
    n_success = sum(1 for row in rows if row["success"])

    # The lines a student (and the smoke test) should be able to find.
    print()
    print(f"Success rate: {success_rate:.1f}%  ({episodes} episodes)")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Average max reward: {avg_max_reward:.3f}")
    print(f"Runtime: {elapsed:.1f}s")
    print()
    print("Per episode:")
    for row in rows:
        print(f"  {format_episode_line(row)}")
    print()
    print(format_stage_report(stages))
    if video_paths:
        print("Videos:")
        for path in video_paths:
            print(f"  {path}")
    else:
        print("Videos: none (pass --videos 1 or more, or --save-failures)")
    print(f"Wrote {info_path}")

    return {
        "checkpoint": policy_path,
        "step": step,
        "episodes": episodes,
        "successes": n_success,
        "success_rate": success_rate,
        "avg_sum_reward": avg_reward,
        "avg_max_reward": avg_max_reward,
        "per_episode": rows,
        "video_paths": video_paths,
        "output_dir": str(output_dir),
        "seed": seed,
        "temporal_ensemble_coeff": None if temporal_ensemble is None else float(temporal_ensemble),
        "cube_sampling": cube_record,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    evaluate_checkpoint(
        args.checkpoint,
        episodes=args.episodes,
        videos=args.videos,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        seed=args.seed,
        save_failures=args.save_failures,
        temporal_ensemble=args.temporal_ensemble,
        cube_range=args.cube_range,
        cube_x=None if args.cube_x is None else (args.cube_x[0], args.cube_x[1]),
        cube_y=None if args.cube_y is None else (args.cube_y[0], args.cube_y[1]),
    )


if __name__ == "__main__":
    main()
