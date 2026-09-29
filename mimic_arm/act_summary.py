"""Print the size of a trained ACT policy, piece by piece.

The names below are the attributes on LeRobot 0.6.1's ``ACT`` module
(``policy.model`` inside ``ACTPolicy``). They are not guesses:

* ``backbone`` is the ResNet, wrapped by torchvision's ``IntermediateLayerGetter``
* ``encoder`` is the transformer encoder (``ACTEncoder``)
* ``decoder`` is the transformer decoder (``ACTDecoder``)
* ``vae_encoder`` is the second transformer, used only while training
* modules whose names end in ``_proj``, plus ``action_head``, are the linear
  (or 1x1 conv) layers that change vector sizes on the way in or out
"""

from __future__ import annotations

import argparse
from pathlib import Path

ENV_TASK = "AlohaTransferCube-v0"

# Keys on ACTConfig. Printed so a reader can see the shape choices, not only
# the parameter totals.
CONFIG_KEYS = (
    "dim_model",
    "n_heads",
    "dim_feedforward",
    "n_encoder_layers",
    "n_decoder_layers",
    "latent_dim",
    "chunk_size",
    "n_action_steps",
)


def _count(module) -> tuple[int, int]:
    """Return ``(total, trainable)`` parameter counts for one module."""
    total = 0
    trainable = 0
    for param in module.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return total, trainable


def _classify(name: str) -> str:
    """Group one child of ``policy.model`` into a block a student can name."""
    if name == "backbone":
        return "backbone"
    if name == "encoder":
        return "encoder"
    if name == "decoder":
        return "decoder"
    if name == "vae_encoder":
        return "vae_encoder"
    if name == "action_head" or name.endswith("_proj"):
        return "projections"
    return "other"


def summarize_act_policy(policy) -> str:
    """Return a multi-line report of parameter counts and ACT config values."""
    model = policy.model
    total, trainable = _count(policy)
    groups: dict[str, list[tuple[str, int, int]]] = {
        "backbone": [],
        "encoder": [],
        "decoder": [],
        "vae_encoder": [],
        "projections": [],
        "other": [],
    }
    for name, child in model.named_children():
        child_total, child_trainable = _count(child)
        groups[_classify(name)].append((name, child_total, child_trainable))

    lines = [
        f"policy class: {type(policy).__name__}",
        f"model class: {type(model).__name__}",
        f"total parameters: {total:,}",
        f"trainable parameters: {trainable:,}",
        "",
    ]

    def _block(title: str, key: str) -> None:
        rows = groups[key]
        block_total = sum(row[1] for row in rows)
        lines.append(f"{title}: {block_total:,}")
        for name, child_total, _child_trainable in rows:
            lines.append(f"  model.{name}: {child_total:,}")

    _block("model.backbone (ResNet backbone)", "backbone")
    _block("model.encoder (transformer encoder)", "encoder")
    _block("model.decoder (transformer decoder)", "decoder")
    _block("model.vae_encoder (VAE encoder)", "vae_encoder")
    _block("input/output projections", "projections")
    _block("other parameters (embeddings and position tables)", "other")

    accounted = sum(row[1] for rows in groups.values() for row in rows)
    lines.append("")
    if accounted == total:
        lines.append("These blocks add up to the total.")
    else:
        lines.append(
            f"Block sum {accounted:,} does not match the total {total:,}. "
            "Some parameters are not direct children of policy.model."
        )

    lines.append("")
    lines.append("config:")
    config = policy.config
    for key in CONFIG_KEYS:
        lines.append(f"  {key}: {getattr(config, key)}")
    return "\n".join(lines)


def load_act_policy(checkpoint: str, device: str = "cpu"):
    """Load an ACT checkpoint the same way ``evaluate.py`` does, on ``device``."""
    from mimic_arm.mujoco_gl import configure_mujoco_rendering

    configure_mujoco_rendering()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import AlohaEnv
    from lerobot.policies import make_policy

    from mimic_arm.checkpoints import resolve_checkpoint

    policy_path = resolve_checkpoint(checkpoint)
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = Path(policy_path)
    policy_cfg.device = device
    # make_policy needs the sim's observation sizes even when we only count
    # parameters. AlohaEnv describes the task; it does not start a rollout.
    env_cfg = AlohaEnv(task=ENV_TASK)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    return policy


def print_act_summary(checkpoint: str, device: str = "cpu") -> str:
    """Load ``checkpoint`` and print the parameter report. Returns the text."""
    policy = load_act_policy(checkpoint, device=device)
    text = summarize_act_policy(policy)
    print(text)
    return text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Checkpoint folder, training directory, or Hub id.")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args(argv)
    print_act_summary(args.checkpoint, device=args.device)


if __name__ == "__main__":
    main()
