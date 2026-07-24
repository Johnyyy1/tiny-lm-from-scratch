from __future__ import annotations

import argparse
import heapq
import math
import os
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

STOP_TOKEN = "^"
CHECKPOINT_VERSION = 1


# =============================================================================
# Configuration
# =============================================================================


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc


@dataclass(frozen=True)
class Config:
    data_file: Path
    vocab_size: int
    seq_len: int
    d_model: int
    num_heads: int
    d_ff: int
    num_layers: int
    batch_size: int
    max_steps: int
    eval_interval: int
    eval_batches: int
    learning_rate: float
    dropout: float
    max_token_length: int
    compile_model: bool
    seed: int = 42

    @classmethod
    def from_env(cls) -> Config:
        config = cls(
            data_file=Path(os.environ.get("MINIBPE_DATA_FILE", "data/input.txt")),
            vocab_size=env_int("MINIBPE_VOCAB_SIZE", 5000),
            seq_len=env_int("MINIBPE_SEQ_LEN", 512),
            d_model=env_int("MINIBPE_D_MODEL", 1024),
            num_heads=env_int("MINIBPE_NUM_HEADS", 8),
            d_ff=env_int("MINIBPE_D_FF", 2048),
            num_layers=env_int("MINIBPE_NUM_LAYERS", 2),
            batch_size=env_int("MINIBPE_BATCH_SIZE", 16),
            max_steps=env_int("MINIBPE_MAX_STEPS", 200000),
            eval_interval=env_int("MINIBPE_EVAL_INTERVAL", 100),
            eval_batches=env_int("MINIBPE_EVAL_BATCHES", 4),
            learning_rate=env_float("MINIBPE_LEARNING_RATE", 3e-4),
            dropout=env_float("MINIBPE_DROPOUT", 0.1),
            max_token_length=env_int("MINIBPE_MAX_TOKEN_LENGTH", 40),
            compile_model=os.environ.get("MINIBPE_COMPILE", "0") == "1",
            seed=env_int("MINIBPE_SEED", 42),
        )
        config.validate()
        return config

    def validate(self) -> None:
        positive_values = {
            "MINIBPE_VOCAB_SIZE": self.vocab_size,
            "MINIBPE_SEQ_LEN": self.seq_len,
            "MINIBPE_D_MODEL": self.d_model,
            "MINIBPE_NUM_HEADS": self.num_heads,
            "MINIBPE_D_FF": self.d_ff,
            "MINIBPE_NUM_LAYERS": self.num_layers,
            "MINIBPE_BATCH_SIZE": self.batch_size,
            "MINIBPE_EVAL_INTERVAL": self.eval_interval,
            "MINIBPE_EVAL_BATCHES": self.eval_batches,
            "MINIBPE_MAX_TOKEN_LENGTH": self.max_token_length,
        }
        for name, value in positive_values.items():
            if value < 1:
                raise ValueError(f"{name} must be positive.")
        if self.max_steps < 0:
            raise ValueError("MINIBPE_MAX_STEPS cannot be negative.")
        if self.d_model % self.num_heads:
            raise ValueError("MINIBPE_NUM_HEADS must divide MINIBPE_D_MODEL.")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("MINIBPE_DROPOUT must be in the range [0, 1).")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("MINIBPE_LEARNING_RATE must be positive.")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["data_file"] = str(self.data_file)
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Config:
        values = dict(values)
        values["data_file"] = Path(values["data_file"])
        config = cls(**values)
        config.validate()
        return config


def load_data(path: Path) -> tuple[list[str], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file {str(path)!r} does not exist.")

    with path.open(encoding="utf-8") as handle:
        data = [line.rstrip("\n") for line in handle]

    if not data or all(not sample for sample in data):
        raise ValueError(f"Dataset file {str(path)!r} is empty.")
    if len(data) < 2:
        raise ValueError("Dataset must contain at least two lines for train/validation splitting.")
    if any(STOP_TOKEN in sample for sample in data):
        raise ValueError(f"Training samples cannot contain the reserved token {STOP_TOKEN!r}.")

    split_index = max(1, min(len(data) - 1, int(len(data) * 0.9)))
    return data[:split_index], data[split_index:]


# =============================================================================
# Tokenizer
# =============================================================================


class BPETokenizer:
    """Byte-pair-encoding tokenizer trained from a character vocabulary."""

    def __init__(
        self,
        token_to_id: dict[str, int],
        merges: Sequence[tuple[int, int, int]] | None = None,
    ):
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {token_id: token for token, token_id in token_to_id.items()}
        self.stop_id = token_to_id[STOP_TOKEN]
        self.merges = list(merges) if merges is not None else self._infer_merges()
        self._merge_ranks = {
            (left_id, right_id): (rank, merged_id)
            for rank, (left_id, right_id, merged_id) in enumerate(self.merges)
        }

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def to_dict(self) -> dict[str, int]:
        """Serialize the token vocabulary."""

        return dict(self.token_to_id)

    @classmethod
    def from_dict(
        cls,
        values: dict[str, int],
        merges: Sequence[Sequence[int]] | None = None,
    ) -> BPETokenizer:
        """Restore a tokenizer from a serialized vocabulary and merge list."""

        token_to_id = {str(token): int(token_id) for token, token_id in values.items()}
        if token_to_id.get(STOP_TOKEN) != 0:
            raise ValueError("Checkpoint tokenizer has an invalid stop token.")
        expected_ids = set(range(len(token_to_id)))
        if set(token_to_id.values()) != expected_ids:
            raise ValueError("Checkpoint tokenizer IDs must be contiguous.")
        parsed_merges = None
        if merges is not None:
            parsed_merges = [tuple(int(value) for value in merge) for merge in merges]
            if any(len(merge) != 3 for merge in parsed_merges):
                raise ValueError("Checkpoint tokenizer has an invalid merge list.")
        return cls(token_to_id, parsed_merges)

    def _infer_merges(self) -> list[tuple[int, int, int]]:
        """Reconstruct merge rules from legacy vocab-only checkpoints."""

        merges: list[tuple[int, int, int]] = []
        ranks: dict[tuple[int, int], tuple[int, int]] = {}
        ordered_tokens = sorted(self.id_to_token.items())
        for merged_id, merged_text in ordered_tokens:
            if len(merged_text) < 2:
                continue
            token_ids = [self.token_to_id[char] for char in merged_text]
            token_ids = self._apply_merges(token_ids, ranks)
            if len(token_ids) != 2:
                raise ValueError(
                    f"Cannot reconstruct merge rule for vocabulary token {merged_text!r}."
                )
            left_id, right_id = token_ids
            merges.append((left_id, right_id, merged_id))
            ranks[(left_id, right_id)] = (len(merges) - 1, merged_id)
        return merges

    @staticmethod
    def _apply_merges(
        token_ids: list[int],
        merge_ranks: dict[tuple[int, int], tuple[int, int]],
    ) -> list[int]:
        while len(token_ids) > 1:
            pairs = (
                (token_ids[index], token_ids[index + 1]) for index in range(len(token_ids) - 1)
            )
            candidates = (
                (merge_ranks[pair][0], pair, merge_ranks[pair][1])
                for pair in pairs
                if pair in merge_ranks
            )
            try:
                _, best_pair, merged_id = min(candidates)
            except ValueError:
                break

            merged: list[int] = []
            index = 0
            while index < len(token_ids):
                if (
                    index + 1 < len(token_ids)
                    and (token_ids[index], token_ids[index + 1]) == best_pair
                ):
                    merged.append(merged_id)
                    index += 2
                else:
                    merged.append(token_ids[index])
                    index += 1
            token_ids = merged
        return token_ids

    @classmethod
    def train(
        cls,
        samples: Sequence[str],
        target_vocab_size: int,
        max_token_length: int,
    ) -> BPETokenizer:
        """Build a BPE vocabulary from the supplied text samples."""

        characters = sorted({char for sample in samples for char in sample})
        if not characters:
            raise ValueError("Cannot train a tokenizer on empty text.")
        token_to_id = {char: index + 1 for index, char in enumerate(characters)}
        token_to_id[STOP_TOKEN] = 0
        id_to_token = {token_id: token for token, token_id in token_to_id.items()}
        merges: list[tuple[int, int, int]] = []
        recorded_pairs = set()

        if target_vocab_size < len(token_to_id):
            raise ValueError(
                "MINIBPE_VOCAB_SIZE must be at least the initial vocabulary "
                f"size ({len(token_to_id)})."
            )

        stop_id = token_to_id[STOP_TOKEN]
        tokens: list[int] = []
        for sample in samples:
            tokens.extend(token_to_id[char] for char in sample)
            tokens.append(stop_id)

        token_count = len(tokens)
        previous = [index - 1 for index in range(token_count)]
        following = [index + 1 for index in range(token_count)]
        if token_count:
            previous[0] = -1
            following[-1] = -1
        alive = bytearray(b"\x01") * token_count

        pair_positions: dict[tuple[int, int], set] = defaultdict(set)
        for left in range(token_count - 1):
            right = following[left]
            if tokens[left] != stop_id and tokens[right] != stop_id:
                pair_positions[(tokens[left], tokens[right])].add(left)

        versions: dict[tuple[int, int], int] = defaultdict(int)
        heap = [
            (-len(positions), 0, pair) for pair, positions in pair_positions.items() if positions
        ]
        heapq.heapify(heap)
        invalid_pairs = set()

        def current_pair(left: int) -> tuple[int, int] | None:
            if left < 0 or not alive[left]:
                return None
            right = following[left]
            if right < 0 or not alive[right]:
                return None
            if tokens[left] == stop_id or tokens[right] == stop_id:
                return None
            return tokens[left], tokens[right]

        def remove_pair(left: int, changed: set) -> None:
            pair = current_pair(left)
            if pair is not None:
                pair_positions[pair].discard(left)
                changed.add(pair)

        def add_pair(left: int, changed: set) -> None:
            pair = current_pair(left)
            if pair is not None:
                pair_positions[pair].add(left)
                changed.add(pair)

        while len(token_to_id) < target_vocab_size and heap:
            while heap:
                negative_count, version, pair = heapq.heappop(heap)
                if (
                    versions[pair] == version
                    and -negative_count == len(pair_positions[pair])
                    and pair_positions[pair]
                ):
                    break
            else:
                break

            merged_text = id_to_token[pair[0]] + id_to_token[pair[1]]
            if len(merged_text) > max_token_length:
                invalid_pairs.add(pair)
                continue

            merged_id = token_to_id.get(merged_text)
            if merged_id is None:
                merged_id = len(token_to_id)
                token_to_id[merged_text] = merged_id
                id_to_token[merged_id] = merged_text
            if pair not in recorded_pairs:
                merges.append((pair[0], pair[1], merged_id))
                recorded_pairs.add(pair)

            changed_pairs = set()
            merged_occurrences = 0
            for left in sorted(pair_positions[pair]):
                if current_pair(left) != pair:
                    continue

                right = following[left]
                before = previous[left]
                after = following[right]

                remove_pair(before, changed_pairs)
                remove_pair(left, changed_pairs)
                remove_pair(right, changed_pairs)

                tokens[left] = merged_id
                following[left] = after
                if after >= 0:
                    previous[after] = left
                alive[right] = 0
                previous[right] = -1
                following[right] = -1

                add_pair(before, changed_pairs)
                add_pair(left, changed_pairs)
                merged_occurrences += 1

            for changed_pair in changed_pairs:
                versions[changed_pair] += 1
                count = len(pair_positions[changed_pair])
                if count and changed_pair not in invalid_pairs:
                    heapq.heappush(
                        heap,
                        (-count, versions[changed_pair], changed_pair),
                    )

            if merged_occurrences == 0:
                continue
        return cls(token_to_id, merges)

    def encode(self, text: str, add_stop: bool = False) -> list[int]:
        """Convert text into token IDs using the learned BPE merge order."""

        token_ids = []
        for position, char in enumerate(text):
            try:
                token_ids.append(self.token_to_id[char])
            except KeyError:
                raise ValueError(
                    f"Character {char!r} at position {position} is not in the tokenizer vocabulary."
                ) from None
        token_ids = self._apply_merges(token_ids, self._merge_ranks)
        if add_stop:
            token_ids.append(self.stop_id)
        return token_ids

    def decode(self, token_ids: Iterable[int]) -> str:
        """Convert token IDs back into text."""

        try:
            return "".join(self.id_to_token[token_id] for token_id in token_ids)
        except KeyError as exc:
            raise ValueError(f"Unknown token ID: {exc.args[0]}") from None


# =============================================================================
# Dataset
# =============================================================================


@dataclass
class EncodedDataset:
    tokens: torch.Tensor
    valid_lens: torch.Tensor

    def __len__(self) -> int:
        return self.tokens.shape[0]


def encode_dataset(
    samples: Sequence[str],
    tokenizer: BPETokenizer,
    max_seq_len: int,
    name: str,
) -> EncodedDataset:
    encoded = []
    discarded = 0
    for sample in samples:
        token_ids = tokenizer.encode(sample, add_stop=True)
        input_length = len(token_ids) - 1
        if input_length < 1 or input_length > max_seq_len:
            discarded += 1
            continue
        encoded.append(token_ids)

    if not encoded:
        raise ValueError(f"No {name} samples fit within MINIBPE_SEQ_LEN.")

    width = max(len(token_ids) for token_ids in encoded)
    tokens = torch.full(
        (len(encoded), width),
        tokenizer.stop_id,
        dtype=torch.long,
    )
    valid_lens = torch.empty(len(encoded), dtype=torch.long)
    for row, token_ids in enumerate(encoded):
        tokens[row, : len(token_ids)] = torch.tensor(token_ids)
        valid_lens[row] = len(token_ids) - 1

    print(
        f"{name.capitalize()} samples: {len(encoded)} "
        f"({discarded} discarded, max input length {width - 1})"
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


# =============================================================================
# Transformer
# =============================================================================


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x).view(
            batch,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )
        q, k, v = qkv.unbind(dim=2)
        return (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )

    def forward(self, x: torch.Tensor, valid_lens: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q, k, v = self._project(x)
        positions = torch.arange(seq_len, device=x.device)
        causal = positions.unsqueeze(0) <= positions.unsqueeze(1)
        valid_keys = positions.unsqueeze(0) < valid_lens.unsqueeze(1)
        attention_mask = causal.view(1, 1, seq_len, seq_len) & valid_keys.view(batch, 1, 1, seq_len)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.output(output)

    def step(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        q, k, v = self._project(x)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        output = F.scaled_dot_product_attention(q, k, v)
        output = output.transpose(1, 2).contiguous().view(x.shape[0], 1, -1)
        return self.output(output), (k, v)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.attention = CausalSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, valid_lens: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attention(x, valid_lens)))
        return self.norm2(x + self.dropout(self.ffn(x)))

    def step(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attention, cache = self.attention.step(x, cache)
        x = self.norm1(x + attention)
        return self.norm2(x + self.ffn(x)), cache


class MiniGPT(nn.Module):
    def __init__(self, config: Config, vocab_size: int):
        super().__init__()
        self.max_seq_len = config.seq_len
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.seq_len, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model,
                    config.d_ff,
                    config.num_heads,
                    config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        valid_lens: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = token_ids.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(f"Input length {seq_len} exceeds model limit {self.max_seq_len}.")
        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)
        for block in self.blocks:
            x = block(x, valid_lens)
        return self.output(self.final_norm(x))

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: Sequence[int],
        stop_id: int,
        temperature: float,
        top_k: int,
        max_new_tokens: int | None = None,
    ) -> list[int]:
        """Generate token IDs autoregressively from a non-empty prompt."""

        if not prompt_ids:
            raise ValueError("The prompt must contain at least one token.")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive.")
        if not 1 <= top_k <= self.output.out_features:
            raise ValueError(f"top_k must be between 1 and {self.output.out_features}.")
        if len(prompt_ids) > self.max_seq_len:
            raise ValueError("The prompt is longer than MINIBPE_SEQ_LEN.")
        if max_new_tokens is not None and max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")

        self.eval()
        device = self.token_embedding.weight.device
        caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [None for _ in self.blocks]
        generated = list(prompt_ids)
        logits = None

        for position, token_id in enumerate(prompt_ids):
            token = torch.tensor([[token_id]], device=device)
            position_id = torch.tensor([position], device=device)
            x = self.token_embedding(token) + self.position_embedding(position_id)
            new_caches = []
            for block, cache in zip(self.blocks, caches):
                x, cache = block.step(x, cache)
                new_caches.append(cache)
            caches = new_caches
            logits = self.output(self.final_norm(x))[0, -1]

        generation_limit = self.max_seq_len
        if max_new_tokens is not None:
            generation_limit = min(
                generation_limit,
                len(prompt_ids) + max_new_tokens,
            )

        while len(generated) < generation_limit:
            top_logits, top_indices = torch.topk(logits, top_k)
            probabilities = F.softmax(top_logits / temperature, dim=-1)
            sampled = torch.multinomial(probabilities, 1).item()
            token_id = top_indices[sampled].item()
            if token_id == stop_id:
                break

            generated.append(token_id)
            position = len(generated) - 1
            token = torch.tensor([[token_id]], device=device)
            position_id = torch.tensor([position], device=device)
            x = self.token_embedding(token) + self.position_embedding(position_id)
            new_caches = []
            for block, cache in zip(self.blocks, caches):
                x, cache = block.step(x, cache)
                new_caches.append(cache)
            caches = new_caches
            logits = self.output(self.final_norm(x))[0, -1]

        return generated


# =============================================================================
# Training
# =============================================================================


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


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.inference_mode()
def estimate_loss(
    model: nn.Module,
    dataset: EncodedDataset,
    config: Config,
    generator: torch.Generator,
    device: torch.device,
) -> float:
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
    optimizer_kwargs = {"lr": config.learning_rate}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    scheduler = create_lr_scheduler(optimizer, config.max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    return optimizer, scheduler, scaler


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


# =============================================================================
# Checkpointing
# =============================================================================


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
    required = {
        "step",
        "config",
        "tokenizer",
        "model_state_dict",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {', '.join(missing)}")
    return checkpoint


def model_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[MiniGPT, BPETokenizer, Config]:
    config = Config.from_dict(checkpoint["config"])
    tokenizer = BPETokenizer.from_dict(
        checkpoint["tokenizer"],
        checkpoint.get("tokenizer_merges"),
    )
    model = MiniGPT(config, tokenizer.vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, tokenizer, config


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

    optimizer, scheduler, scaler = create_training_components(
        model,
        config,
        device,
    )
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
            scheduler = create_lr_scheduler(
                optimizer,
                config.max_steps - start_step,
            )
            print(
                "Restarting the learning-rate schedule for "
                f"{config.max_steps - start_step} additional steps"
            )
        training_losses = list(resume_state.get("training_losses", []))
        validation_losses = list(resume_state.get("validation_losses", []))
        restore_rng_state(
            resume_state["rng_state"],
            train_generator,
            eval_generator,
        )
        print(f"Resuming from step {start_step}")

    forward_model = model
    if config.compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("MINIBPE_COMPILE=1 requires torch.compile support.")
        forward_model = torch.compile(model)

    model.train()
    last_saved_step = -1
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
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        training_losses.append(loss.item())

        should_evaluate = (
            step == 0 or (step + 1) % config.eval_interval == 0 or step + 1 == config.max_steps
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
            print(
                f"{step + 1:7d}/{config.max_steps}: "
                f"train_loss={training_losses[-1]:.4f} "
                f"val_loss={validation_loss:.4f}"
            )

        completed_step = step + 1
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


# =============================================================================
# Evaluation and generation
# =============================================================================


@torch.inference_mode()
def perplexity(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    samples: Sequence[str],
    device: torch.device,
) -> float:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for sample in samples:
        token_ids = tokenizer.encode(sample, add_stop=True)
        if not 2 <= len(token_ids) <= model.max_seq_len + 1:
            continue
        inputs = torch.tensor([token_ids[:-1]], device=device)
        targets = torch.tensor([token_ids[1:]], device=device)
        valid_lens = torch.tensor([inputs.shape[1]], device=device)
        logits = model(inputs, valid_lens)
        total_nll += F.cross_entropy(
            logits[0],
            targets[0],
            reduction="sum",
        ).item()
        total_tokens += inputs.shape[1]

    if total_tokens == 0:
        raise ValueError("No evaluation samples contain usable token sequences.")
    return math.exp(total_nll / total_tokens)


@torch.inference_mode()
def dataset_metrics(
    model: MiniGPT,
    dataset: EncodedDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
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


# =============================================================================
# CLI
# =============================================================================


CONFIG_ARGUMENTS = {
    "data_file": ("--data-file", Path),
    "vocab_size": ("--vocab-size", int),
    "seq_len": ("--seq-len", int),
    "d_model": ("--d-model", int),
    "num_heads": ("--num-heads", int),
    "d_ff": ("--d-ff", int),
    "num_layers": ("--num-layers", int),
    "batch_size": ("--batch-size", int),
    "max_steps": ("--max-steps", int),
    "eval_interval": ("--eval-interval", int),
    "eval_batches": ("--eval-batches", int),
    "learning_rate": ("--learning-rate", float),
    "dropout": ("--dropout", float),
    "max_token_length": ("--max-token-length", int),
    "seed": ("--seed", int),
}


def add_device_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="execution device (default: auto)",
    )


def add_config_arguments(
    parser: argparse.ArgumentParser,
    fields: Sequence[str] | None = None,
) -> None:
    selected = fields or tuple(CONFIG_ARGUMENTS)
    for field in selected:
        option, value_type = CONFIG_ARGUMENTS[field]
        parser.add_argument(
            option,
            dest=field,
            type=value_type,
            default=None,
        )
    parser.add_argument(
        "--compile",
        dest="compile_model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable torch.compile",
    )


def config_with_cli_overrides(
    config: Config,
    args: argparse.Namespace,
) -> Config:
    updates = {}
    for field in (*CONFIG_ARGUMENTS, "compile_model"):
        if hasattr(args, field):
            value = getattr(args, field)
            if value is not None:
                updates[field] = value
    updated = replace(config, **updates)
    updated.validate()
    return updated


def initialize_runtime(seed: int, requested_device: str) -> torch.device:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(requested_device)
    print(f"Device: {device}")
    return device


def run_train(args: argparse.Namespace) -> None:
    config = config_with_cli_overrides(Config.from_env(), args)
    train_samples, validation_samples = load_data(config.data_file)
    device = initialize_runtime(config.seed, args.device)
    tokenizer = BPETokenizer.train(
        train_samples,
        config.vocab_size,
        config.max_token_length,
    )
    train_data = encode_dataset(
        train_samples,
        tokenizer,
        config.seq_len,
        "training",
    )
    validation_data = encode_dataset(
        validation_samples,
        tokenizer,
        config.seq_len,
        "validation",
    )

    model = MiniGPT(config, tokenizer.vocab_size).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Transformer parameters: {parameter_count:,}")
    train_model(
        model,
        tokenizer,
        train_data,
        validation_data,
        config,
        device,
        checkpoint_path=args.checkpoint,
        checkpoint_interval=args.checkpoint_interval,
    )


def run_resume(args: argparse.Namespace) -> None:
    checkpoint = load_checkpoint(args.checkpoint)
    checkpoint_config = Config.from_dict(checkpoint["config"])
    config = config_with_cli_overrides(checkpoint_config, args)
    completed_step = int(checkpoint["step"])
    if config.max_steps <= completed_step:
        raise ValueError(
            f"--max-steps must exceed checkpoint step {completed_step}; "
            f"received {config.max_steps}."
        )

    device = initialize_runtime(config.seed, args.device)
    tokenizer = BPETokenizer.from_dict(
        checkpoint["tokenizer"],
        checkpoint.get("tokenizer_merges"),
    )
    train_samples, validation_samples = load_data(config.data_file)
    train_data = encode_dataset(
        train_samples,
        tokenizer,
        config.seq_len,
        "training",
    )
    validation_data = encode_dataset(
        validation_samples,
        tokenizer,
        config.seq_len,
        "validation",
    )

    model = MiniGPT(config, tokenizer.vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    output_checkpoint = args.output_checkpoint or args.checkpoint
    train_model(
        model,
        tokenizer,
        train_data,
        validation_data,
        config,
        device,
        checkpoint_path=output_checkpoint,
        checkpoint_interval=args.checkpoint_interval,
        resume_state=checkpoint,
    )


def run_generate(args: argparse.Namespace) -> None:
    checkpoint = load_checkpoint(args.checkpoint)
    config = Config.from_dict(checkpoint["config"])
    seed = config.seed if args.seed is None else args.seed
    device = initialize_runtime(seed, args.device)
    model, tokenizer, _ = model_from_checkpoint(checkpoint, device)
    prompt_ids = tokenizer.encode(args.prompt)
    generated_ids = model.generate(
        prompt_ids,
        tokenizer.stop_id,
        temperature=args.temperature,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )
    print(tokenizer.decode(generated_ids))


def run_evaluate(args: argparse.Namespace) -> None:
    checkpoint = load_checkpoint(args.checkpoint)
    checkpoint_config = Config.from_dict(checkpoint["config"])
    data_file = args.data_file or checkpoint_config.data_file
    device = initialize_runtime(checkpoint_config.seed, args.device)
    model, tokenizer, _ = model_from_checkpoint(checkpoint, device)
    train_samples, validation_samples = load_data(data_file)
    samples = train_samples if args.split == "train" else validation_samples
    dataset = encode_dataset(
        samples,
        tokenizer,
        checkpoint_config.seq_len,
        args.split,
    )
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive.")
        dataset = EncodedDataset(
            dataset.tokens[: args.max_samples],
            dataset.valid_lens[: args.max_samples],
        )
    batch_size = args.batch_size or checkpoint_config.batch_size
    if batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    loss, result = dataset_metrics(
        model,
        dataset,
        batch_size,
        device,
    )
    print(f"{args.split}_loss={loss:.6f}")
    print(f"{args.split}_perplexity={result:.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and use the mini BPE transformer.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="train a new tokenizer and model",
    )
    add_device_argument(train_parser)
    add_config_arguments(train_parser)
    train_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/latest.pt"),
        help="checkpoint output path (default: checkpoints/latest.pt)",
    )
    train_parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="save every N completed steps (default: 1000)",
    )
    train_parser.set_defaults(handler=run_train)

    resume_parser = subparsers.add_parser(
        "resume",
        help="resume training from a checkpoint",
    )
    resume_parser.add_argument("checkpoint", type=Path)
    add_device_argument(resume_parser)
    add_config_arguments(
        resume_parser,
        fields=(
            "data_file",
            "batch_size",
            "max_steps",
            "eval_interval",
            "eval_batches",
        ),
    )
    resume_parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=None,
        help="write to a new checkpoint instead of replacing the input",
    )
    resume_parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="save every N completed steps (default: 1000)",
    )
    resume_parser.set_defaults(handler=run_resume)

    generate_parser = subparsers.add_parser(
        "generate",
        help="generate text from a checkpoint",
    )
    generate_parser.add_argument("checkpoint", type=Path)
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--temperature", type=float, default=1.0)
    generate_parser.add_argument("--top-k", type=int, default=50)
    generate_parser.add_argument("--max-new-tokens", type=int, default=100)
    generate_parser.add_argument("--seed", type=int, default=None)
    add_device_argument(generate_parser)
    generate_parser.set_defaults(handler=run_generate)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="evaluate a checkpoint on the configured dataset",
    )
    evaluate_parser.add_argument("checkpoint", type=Path)
    evaluate_parser.add_argument("--data-file", type=Path, default=None)
    evaluate_parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="validation",
    )
    evaluate_parser.add_argument("--batch-size", type=int, default=None)
    evaluate_parser.add_argument("--max-samples", type=int, default=None)
    add_device_argument(evaluate_parser)
    evaluate_parser.set_defaults(handler=run_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
