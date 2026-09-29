#!/usr/bin/env python
"""Score several ACT checkpoints on the same simulated episodes.

Pass a training output directory to score every numbered checkpoint in it,
or pass an explicit list. Every checkpoint sees the same seeds and the same
number of episodes, so the table is a fair comparison of training steps.

20 to 50 episodes is a small sample. A success rate of 65% on 20 episodes
can easily be 40% or 85% on the next 20. Each row prints a 95% Wilson
interval, which is a range that likely contains the true success rate.

Examples
--------
Every numbered checkpoint from a training run, 20 episodes each::

    python compare.py --train-dir outputs/train/act_aloha_transfer_cube

Two specific checkpoints, same three episodes::

    python compare.py \\
        --checkpoints outputs/train/smoke/checkpoints/000010 \\
                      outputs/train/smoke/checkpoints/000020 \\
        --episodes 3 --device cpu
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import evaluate_checkpoint
from mimic_arm.checkpoints import checkpoint_step, list_numbered_checkpoints, resolve_checkpoint


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for ``successes`` out of ``n`` trials.

    Returns the low and high ends as fractions between 0 and 1. ``z=1.96``
    is the usual cutoff for 95%.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if successes < 0 or successes > n:
        raise ValueError("successes must be between 0 and n")
    proportion = successes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (proportion + z2 / (2.0 * n)) / denominator
    margin = z * math.sqrt((proportion * (1.0 - proportion) / n) + z2 / (4.0 * n * n))
    margin /= denominator
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return low, high


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        help="Training output directory. Every numbered checkpoint folder is evaluated.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        help="Checkpoint paths or Hub ids to compare, instead of --train-dir.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Episodes per checkpoint. The same seeds are reused. Default: 20.",
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
        default=Path("outputs/eval/compare"),
        help="Where compare.csv and each checkpoint's eval_info.json are written.",
    )
    parser.add_argument("--seed", type=int, default=1000, help="Seed of the first episode. Default: 1000.")
    return parser


def _checkpoint_list(args: argparse.Namespace) -> list[str]:
    if args.train_dir and args.checkpoints:
        raise SystemExit("Pass either --train-dir or --checkpoints, not both.")
    if args.train_dir:
        return [str(path) for path in list_numbered_checkpoints(args.train_dir)]
    if args.checkpoints:
        return list(args.checkpoints)
    raise SystemExit("Pass --train-dir or --checkpoints. See compare.py --help.")


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "step",
        "checkpoint",
        "episodes",
        "successes",
        "success_rate_percent",
        "avg_max_reward",
        "avg_sum_reward",
        "wilson95_low_percent",
        "wilson95_high_percent",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _print_table(rows: list[dict]) -> None:
    print()
    print(
        "Note: 20-50 episodes is a small sample, so these success rates have wide "
        "error bars. The 95% Wilson interval is the range that likely contains the "
        "true success rate. A few extra successes can move a row a lot."
    )
    print()
    header = (
        f"{'step':>8}  {'success rate':>13}  {'avg max reward':>15}  {'95% Wilson interval':>22}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        step = "n/a" if row["step"] == "" else str(row["step"])
        print(
            f"{step:>8}  {row['success_rate_percent']:12.1f}%  "
            f"{row['avg_max_reward']:15.3f}  "
            f"[{row['wilson95_low_percent']:5.1f}%, {row['wilson95_high_percent']:5.1f}%]"
        )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1.")

    checkpoints = _checkpoint_list(args)
    last_seed = args.seed + args.episodes - 1
    print(
        f"Comparing {len(checkpoints)} checkpoint(s) on {args.episodes} episodes "
        f"(seeds {args.seed} through {last_seed})."
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_rows: list[dict] = []
    for index, checkpoint in enumerate(checkpoints):
        policy_path = resolve_checkpoint(checkpoint)
        step = checkpoint_step(policy_path)
        label = f"step {step}" if step is not None else policy_path
        folder_name = f"step_{step:06d}" if step is not None else f"checkpoint_{index:02d}"
        print()
        print(f"=== {label} ===")
        summary = evaluate_checkpoint(
            checkpoint,
            episodes=args.episodes,
            videos=0,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=args.output_dir / folder_name,
            seed=args.seed,
            save_failures=False,
        )
        low, high = wilson_interval(summary["successes"], summary["episodes"])
        table_rows.append(
            {
                "step": "" if summary["step"] is None else summary["step"],
                "checkpoint": summary["checkpoint"],
                "episodes": summary["episodes"],
                "successes": summary["successes"],
                "success_rate_percent": summary["success_rate"],
                "avg_max_reward": summary["avg_max_reward"],
                "avg_sum_reward": summary["avg_sum_reward"],
                "wilson95_low_percent": low * 100.0,
                "wilson95_high_percent": high * 100.0,
                "_sort": summary["step"] if summary["step"] is not None else 10**12,
            }
        )

    table_rows.sort(key=lambda row: row["_sort"])
    _print_table(table_rows)
    csv_path = args.output_dir / "compare.csv"
    _write_csv(csv_path, table_rows)
    print()
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
