"""Checkpoint serialization and restoration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from tiny_lm.config import Config
from tiny_lm.model import MiniGPT
from tiny_lm.tokenizer import BPETokenizer

CHECKPOINT_VERSION = 2


def capture_rng_state(
    train_generator: torch.Generator,
    eval_generator: torch.Generator,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "train_generator": train_generator.get_state(),
        "eval_generator": eval_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if (
        torch.backends.mps.is_available()
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
    ):
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(
    state: dict[str, Any],
    train_generator: torch.Generator,
    eval_generator: torch.Generator,
) -> None:
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "train_generator" in state:
        train_generator.set_state(state["train_generator"])
    elif "batch_generator" in state:
        train_generator.set_state(state["batch_generator"])
    if "eval_generator" in state:
        eval_generator.set_state(state["eval_generator"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if (
        "mps" in state
        and torch.backends.mps.is_available()
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "set_rng_state")
    ):
        torch.mps.set_rng_state(state["mps"])


def save_checkpoint(
    path: Path,
    model: MiniGPT,
    tokenizer: BPETokenizer,
    config: Config,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    step: int,
    training_losses: Sequence[float],
    validation_losses: Sequence[float],
    train_generator: torch.Generator,
    eval_generator: torch.Generator,
) -> None:
    """Atomically save all state required to resume training."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "step": step,
        "config": config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "tokenizer_merges": tokenizer.merges,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "training_losses": list(training_losses),
        "validation_losses": list(validation_losses),
        "rng_state": capture_rng_state(train_generator, eval_generator),
    }
    torch.save(payload, temporary_path)
    temporary_path.replace(path)
    print(f"Checkpoint saved: {path} (step {step})")


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load and validate a training checkpoint."""

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version: {checkpoint.get('checkpoint_version')!r}"
        )
    required = {"step", "config", "tokenizer", "model_state_dict"}
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {', '.join(missing)}")
    return checkpoint


def model_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[MiniGPT, BPETokenizer, Config]:
    """Restore a ready-to-use model and tokenizer."""

    config = Config.from_dict(checkpoint["config"])
    tokenizer = BPETokenizer.from_dict(
        checkpoint["tokenizer"],
        checkpoint.get("tokenizer_merges"),
    )
    model = MiniGPT(config, tokenizer.vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, tokenizer, config
