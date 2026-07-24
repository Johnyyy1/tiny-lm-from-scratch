"""Dataset loading, token packing, device selection, and batching."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from tiny_lm.tokenizer import BPETokenizer


@dataclass
class EncodedDataset:
    tokens: torch.Tensor
    valid_lens: torch.Tensor

    def __len__(self) -> int:
        return self.tokens.shape[0]


def load_data(path: Path) -> tuple[list[str], list[str]]:
    """Read newline-separated samples and create a deterministic 90/10 split."""

    if not path.is_file():
        raise FileNotFoundError(f"Dataset file {str(path)!r} does not exist.")
    with path.open(encoding="utf-8") as handle:
        data = [line.rstrip("\n") for line in handle]

    if not data or all(not sample for sample in data):
        raise ValueError(f"Dataset file {str(path)!r} is empty.")
    if len(data) < 2:
        raise ValueError("Dataset must contain at least two lines for train/validation splitting.")
    split_index = max(1, min(len(data) - 1, int(len(data) * 0.9)))
    return data[:split_index], data[split_index:]


def encode_dataset(
    samples: Sequence[str],
    tokenizer: BPETokenizer,
    max_seq_len: int,
    name: str,
) -> EncodedDataset:
    """Pack a complete token stream into fixed-length next-token windows."""

    token_stream: list[int] = []
    for sample in samples:
        token_stream.extend(tokenizer.encode(sample, add_stop=True))
    if len(token_stream) < 2:
        raise ValueError(f"The {name} split does not contain enough tokens.")

    encoded = [
        token_stream[start : start + max_seq_len + 1]
        for start in range(0, len(token_stream) - 1, max_seq_len)
    ]
    width = max(len(window) for window in encoded)
    tokens = torch.full((len(encoded), width), tokenizer.stop_id, dtype=torch.long)
    valid_lens = torch.empty(len(encoded), dtype=torch.long)
    for row, token_ids in enumerate(encoded):
        tokens[row, : len(token_ids)] = torch.tensor(token_ids)
        valid_lens[row] = len(token_ids) - 1

    print(
        f"{name.capitalize()} tokens: {len(token_stream):,} "
        f"({len(encoded):,} packed windows, max input length {width - 1})"
    )
    return EncodedDataset(tokens, valid_lens)


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(requested)


def sample_batch(
    dataset: EncodedDataset,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = torch.randint(len(dataset), (batch_size,), generator=generator)
    valid_lens = dataset.valid_lens[indices]
    width = int(valid_lens.max().item())
    selected = dataset.tokens[indices, : width + 1]
    inputs = selected[:, :-1].to(device, non_blocking=device.type == "cuda")
    targets = selected[:, 1:].to(device, non_blocking=device.type == "cuda")
    valid_lens = valid_lens.to(device, non_blocking=device.type == "cuda")
    return inputs, targets, valid_lens
