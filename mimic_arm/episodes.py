"""Train on a list of episodes without changing normalization.

LeRobot 0.6.1's ``DatasetConfig.episodes`` is forwarded to
``LeRobotDataset`` and selects which episodes the reader yields. The
normalizer is a different object: ``lerobot_train`` passes
``dataset.meta.stats``, and that attribute is loaded from ``meta/stats.json``
for the whole dataset. Filtering episodes does not recompute it. Both
20-demo runs in issue #12 therefore share the full-dataset statistics.
"""

from __future__ import annotations

import json
from pathlib import Path

FULL_DATASET_NORMALIZATION = (
    "Normalization uses the full dataset stats in meta/stats.json. "
    "LeRobot 0.6.1 loads those stats for the whole dataset even when "
    "dataset.episodes limits which demonstrations are sampled."
)


def format_dataset_episodes_flag(episodes: list[int]) -> str:
    """One ``lerobot-train`` token. Draccus accepts ``--dataset.episodes=[0,1]``."""
    if not episodes:
        raise ValueError("episode list is empty.")
    cleaned = [int(episode) for episode in episodes]
    if any(episode < 0 for episode in cleaned):
        raise ValueError(f"episode indices must be non-negative, got {cleaned}.")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"episode indices contain duplicates: {cleaned}.")
    body = ",".join(str(episode) for episode in cleaned)
    return f"--dataset.episodes=[{body}]"


def _flag_present(extra: list[str], name: str) -> bool:
    prefix = name + "="
    return any(token == name or token.startswith(prefix) for token in extra)


def apply_episode_filter(command: list[str], extra: list[str], episodes: list[int]) -> list[str]:
    """Append the LeRobot episode filter. Do not add a stats-recompute flag."""
    if _flag_present(extra, "--dataset.episodes"):
        raise SystemExit("Pass either --episodes or --dataset.episodes, not both.")
    return [*command, format_dataset_episodes_flag(episodes)]


def _as_index_list(value, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise SystemExit(f"{label} must be a non-empty list of episode indices.")
    try:
        indices = [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{label} must be integers.") from exc
    return indices


def parse_episodes_argument(value: str, episode_set: str | None) -> list[int]:
    """Parse ``--episodes``.

    A comma-separated list (``28,42,25``) is used as-is. An existing file may
    be a JSON list, ``{"episodes": [...]}``, or the pick_demos.py object with
    ``alike`` and ``random``. The object needs ``--episode-set``.
    """
    text = value.strip()
    path = Path(text)
    if path.is_file():
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            if episode_set is not None:
                raise SystemExit("--episode-set is only for a JSON object with alike and random lists.")
            return _as_index_list(payload, "episode file")
        if not isinstance(payload, dict):
            raise SystemExit(f"episode file {path} must be a JSON list or object.")
        if "alike" in payload or "random" in payload:
            if episode_set not in ("alike", "random"):
                raise SystemExit(
                    "This file has alike and random lists. Pass --episode-set alike or --episode-set random."
                )
            return _as_index_list(payload[episode_set], episode_set)
        if "episodes" in payload:
            if episode_set is not None:
                raise SystemExit("--episode-set does not apply to a file that only has an episodes list.")
            return _as_index_list(payload["episodes"], "episodes")
        raise SystemExit(f"episode file {path} has no episodes, alike, or random list.")

    if episode_set is not None:
        raise SystemExit("--episode-set is only used when --episodes is a JSON file with alike and random lists.")
    if not text:
        raise SystemExit("--episodes is empty.")
    parts = [part.strip() for part in text.split(",") if part.strip()]
    return _as_index_list(parts, "--episodes")
