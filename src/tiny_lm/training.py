"""Training loop, optimizer, learning-rate schedule, and runtime setup."""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tiny_lm.checkpoint import restore_rng_state, save_checkpoint
from tiny_lm.config import Config
from tiny_lm.data import EncodedDataset, choose_device, sample_batch
from tiny_lm.evaluation import autocast_context, estimate_loss, masked_cross_entropy
from tiny_lm.model import MiniGPT
from tiny_lm.tokenizer import BPETokenizer


def initialize_runtime(seed: int, requested_device: str = "auto") -> torch.device:
    """Seed supported backends and select the execution device."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(requested_device)
    print(f"Device: {device}")
    return device


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
):
    warmup_steps = max(1, total_steps * 2 // 100)

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (1 / 1000) + (1 - 1 / 1000) * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


def create_training_components(
    model: MiniGPT,
    config: Config,
    device: torch.device,
):
    optimizer_kwargs: dict[str, Any] = {
        "lr": config.learning_rate,
        "betas": (0.9, 0.95),
        "weight_decay": 0.1,
    }
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    scheduler = create_lr_scheduler(optimizer, config.max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    return optimizer, scheduler, scaler


def train_model(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    train_data: EncodedDataset,
    validation_data: EncodedDataset,
    config: Config,
    device: torch.device,
    checkpoint_path: Path | None = None,
    checkpoint_interval: int = 1000,
    resume_state: dict[str, Any] | None = None,
) -> tuple[list[float], list[float]]:
    """Train a model and optionally write resumable checkpoints."""

    optimizer, scheduler, scaler = create_training_components(model, config, device)
    train_generator = torch.Generator().manual_seed(config.seed)
    eval_generator = torch.Generator().manual_seed(config.seed + 1)
    training_losses: list[float] = []
    validation_losses: list[float] = []
    start_step = 0

    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive.")
    if resume_state is not None:
        resumable_fields = {
            "optimizer_state_dict",
            "scheduler_state_dict",
            "scaler_state_dict",
            "rng_state",
        }
        missing = sorted(resumable_fields - resume_state.keys())
        if missing:
            raise ValueError(
                "Checkpoint cannot resume training; missing fields: " + ", ".join(missing)
            )
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scaler.load_state_dict(resume_state["scaler_state_dict"])
        start_step = int(resume_state["step"])
        previous_max_steps = Config.from_dict(resume_state["config"]).max_steps
        if config.max_steps == previous_max_steps:
            scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        else:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = config.learning_rate
                parameter_group["initial_lr"] = config.learning_rate
            scheduler = create_lr_scheduler(optimizer, config.max_steps - start_step)
            print(
                "Restarting the learning-rate schedule for "
                f"{config.max_steps - start_step} additional steps"
            )
        training_losses = list(resume_state.get("training_losses", []))
        validation_losses = list(resume_state.get("validation_losses", []))
        restore_rng_state(resume_state["rng_state"], train_generator, eval_generator)
        print(f"Resuming from step {start_step}")

    forward_model = model
    if config.compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("MINIBPE_COMPILE=1 requires torch.compile support.")
        forward_model = torch.compile(model)

    model.train()
    last_saved_step = -1
    training_started = time.perf_counter()
    for step in range(start_step, config.max_steps):
        inputs, targets, valid_lens = sample_batch(
            train_data,
            config.batch_size,
            train_generator,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            logits = forward_model(inputs, valid_lens)
            loss = masked_cross_entropy(logits, targets, valid_lens)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        training_losses.append(loss.item())
        completed_step = step + 1

        should_evaluate = (
            step == 0
            or completed_step % config.eval_interval == 0
            or completed_step == config.max_steps
        )
        if should_evaluate:
            validation_loss = estimate_loss(
                forward_model,
                validation_data,
                config,
                eval_generator,
                device,
            )
            validation_losses.append(validation_loss)
            elapsed = time.perf_counter() - training_started
            session_steps = completed_step - start_step
            remaining = config.max_steps - completed_step
            eta = elapsed / session_steps * remaining
            print(
                f"{completed_step:7d}/{config.max_steps}: "
                f"train_loss={training_losses[-1]:.4f} "
                f"val_loss={validation_loss:.4f} "
                f"grad_norm={float(gradient_norm):.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
            )

        if checkpoint_path is not None and completed_step % checkpoint_interval == 0:
            save_checkpoint(
                checkpoint_path,
                model,
                tokenizer,
                config,
                optimizer,
                scheduler,
                scaler,
                completed_step,
                training_losses,
                validation_losses,
                train_generator,
                eval_generator,
            )
            last_saved_step = completed_step

    if checkpoint_path is not None and last_saved_step != config.max_steps:
        save_checkpoint(
            checkpoint_path,
            model,
            tokenizer,
            config,
            optimizer,
            scheduler,
            scaler,
            config.max_steps,
            training_losses,
            validation_losses,
            train_generator,
            eval_generator,
        )
    return training_losses, validation_losses
