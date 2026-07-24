"""Loss, perplexity, and validation helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

from tiny_lm.config import Config
from tiny_lm.data import EncodedDataset, encode_dataset, sample_batch
from tiny_lm.model import MiniGPT
from tiny_lm.tokenizer import BPETokenizer


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_lens: torch.Tensor,
) -> torch.Tensor:
    batch, seq_len, vocab_size = logits.shape
    losses = F.cross_entropy(
        logits.reshape(batch * seq_len, vocab_size),
        targets.reshape(batch * seq_len),
        reduction="none",
    ).view(batch, seq_len)
    mask = torch.arange(seq_len, device=logits.device).unsqueeze(0) < valid_lens.unsqueeze(1)
    return (losses * mask).sum() / mask.sum()


@torch.inference_mode()
def estimate_loss(
    model: nn.Module,
    dataset: EncodedDataset,
    config: Config,
    generator: torch.Generator,
    device: torch.device,
) -> float:
    """Estimate validation loss without consuming the training RNG."""

    model.eval()
    losses = []
    for _ in range(config.eval_batches):
        inputs, targets, valid_lens = sample_batch(
            dataset,
            config.batch_size,
            generator,
            device,
        )
        with autocast_context(device):
            logits = model(inputs, valid_lens)
            loss = masked_cross_entropy(logits, targets, valid_lens)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.inference_mode()
def dataset_metrics(
    model: MiniGPT,
    dataset: EncodedDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """Calculate exact average loss and perplexity over every usable token."""

    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        valid_lens = dataset.valid_lens[start:end]
        width = int(valid_lens.max().item())
        selected = dataset.tokens[start:end, : width + 1]
        inputs = selected[:, :-1].to(device)
        targets = selected[:, 1:].to(device)
        valid_lens = valid_lens.to(device)

        with autocast_context(device):
            logits = model(inputs, valid_lens)
            batch, seq_len, vocab_size = logits.shape
            losses = F.cross_entropy(
                logits.reshape(batch * seq_len, vocab_size),
                targets.reshape(batch * seq_len),
                reduction="none",
            ).view(batch, seq_len)
            mask = torch.arange(seq_len, device=device).unsqueeze(0) < valid_lens.unsqueeze(1)
            total_nll += (losses * mask).sum().item()
            total_tokens += mask.sum().item()

    if total_tokens == 0:
        raise ValueError("The evaluation dataset contains no usable tokens.")
    average_loss = total_nll / total_tokens
    return average_loss, math.exp(average_loss)


@torch.inference_mode()
def perplexity(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    samples: Sequence[str],
    device: torch.device,
) -> float:
    dataset = encode_dataset(samples, tokenizer, model.max_seq_len, "evaluation")
    _, result = dataset_metrics(model, dataset, batch_size=1, device=device)
    return result
