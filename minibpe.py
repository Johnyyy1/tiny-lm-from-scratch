import heapq
import math
import os
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


STOP_TOKEN = "^"


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
    def from_env(cls) -> "Config":
        config = cls(
            data_file=Path(
                os.environ.get(
                    "MINIBPE_DATA_FILE",
                    "/Users/jonas/Downloads/training_data.txt",
                )
            ),
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
        if not 0 <= self.dropout < 1:
            raise ValueError("MINIBPE_DROPOUT must be in the range [0, 1).")
        if self.learning_rate <= 0:
            raise ValueError("MINIBPE_LEARNING_RATE must be positive.")


def load_data(path: Path) -> Tuple[List[str], List[str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Training data not found at {path}. "
            "Set MINIBPE_DATA_FILE to a UTF-8 text file with one sample per line."
        )

    with path.open(encoding="utf-8") as handle:
        data = [line.rstrip("\n") for line in handle]

    if len(data) < 2:
        raise ValueError("Training data must contain at least two lines.")
    if any(STOP_TOKEN in sample for sample in data):
        raise ValueError(f"Training samples cannot contain the reserved token {STOP_TOKEN!r}.")

    split_index = max(1, min(len(data) - 1, int(len(data) * 0.9)))
    return data[:split_index], data[split_index:]


class BPETokenizer:
    """Greedy BPE tokenizer with incremental pair-frequency training."""

    _TOKEN_ID = object()

    def __init__(self, token_to_id: Dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {token_id: token for token, token_id in token_to_id.items()}
        self.stop_id = token_to_id[STOP_TOKEN]
        self._trie: Dict[object, object] = {}

        for token, token_id in token_to_id.items():
            node = self._trie
            for char in token:
                node = node.setdefault(char, {})
            node[self._TOKEN_ID] = token_id

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @classmethod
    def train(
        cls,
        samples: Sequence[str],
        target_vocab_size: int,
        max_token_length: int,
    ) -> "BPETokenizer":
        characters = sorted({char for sample in samples for char in sample})
        token_to_id = {char: index + 1 for index, char in enumerate(characters)}
        token_to_id[STOP_TOKEN] = 0
        id_to_token = {token_id: token for token, token_id in token_to_id.items()}

        if target_vocab_size < len(token_to_id):
            raise ValueError(
                "MINIBPE_VOCAB_SIZE must be at least the initial vocabulary "
                f"size ({len(token_to_id)})."
            )

        stop_id = token_to_id[STOP_TOKEN]
        tokens: List[int] = []
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

        pair_positions: Dict[Tuple[int, int], set] = defaultdict(set)
        for left in range(token_count - 1):
            right = following[left]
            if tokens[left] != stop_id and tokens[right] != stop_id:
                pair_positions[(tokens[left], tokens[right])].add(left)

        versions: Dict[Tuple[int, int], int] = defaultdict(int)
        heap = [
            (-len(positions), 0, pair)
            for pair, positions in pair_positions.items()
            if positions
        ]
        heapq.heapify(heap)

        print(f"Initial vocabulary size: {len(token_to_id)}")
        print(f"Initial token count: {token_count}")

        def current_pair(left: int) -> Optional[Tuple[int, int]]:
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
                print(
                    f"Most frequent merge exceeds {max_token_length} characters "
                    "(stopping)"
                )
                break

            merged_id = token_to_id.get(merged_text)
            if merged_id is None:
                merged_id = len(token_to_id)
                token_to_id[merged_text] = merged_id
                id_to_token[merged_id] = merged_text

            changed_pairs = set()
            merged_occurrences = 0
            for left in sorted(tuple(pair_positions[pair])):
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
                if count:
                    heapq.heappush(
                        heap,
                        (-count, versions[changed_pair], changed_pair),
                    )

            if merged_occurrences == 0:
                continue
            if len(token_to_id) % 100 == 0:
                remaining = sum(alive)
                reduction = (token_count - remaining) / max(1, token_count)
                print(
                    f"Vocabulary: {len(token_to_id)} "
                    f"({reduction:.1%} token reduction)"
                )

        print(f"Final vocabulary size: {len(token_to_id)}")
        return cls(token_to_id)

    def encode(self, text: str, add_stop: bool = False) -> List[int]:
        if add_stop:
            text += STOP_TOKEN

        result: List[int] = []
        cursor = 0
        while cursor < len(text):
            node = self._trie
            scan = cursor
            best_id = None
            best_end = cursor

            while scan < len(text) and text[scan] in node:
                node = node[text[scan]]
                scan += 1
                token_id = node.get(self._TOKEN_ID)
                if token_id is not None:
                    best_id = token_id
                    best_end = scan

            if best_id is None:
                raise ValueError(
                    f"No vocabulary token matches character {text[cursor]!r} "
                    f"at position {cursor}."
                )
            result.append(best_id)
            cursor = best_end

        return result

    def decode(self, token_ids: Iterable[int]) -> str:
        return "".join(self.id_to_token[token_id] for token_id in token_ids)


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


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_batch(
    dataset: EncodedDataset,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = torch.randint(len(dataset), (batch_size,), generator=generator)
    valid_lens = dataset.valid_lens[indices]
    width = int(valid_lens.max().item())
    selected = dataset.tokens[indices, : width + 1]
    inputs = selected[:, :-1].to(device, non_blocking=device.type == "cuda")
    targets = selected[:, 1:].to(device, non_blocking=device.type == "cuda")
    valid_lens = valid_lens.to(device, non_blocking=device.type == "cuda")
    return inputs, targets, valid_lens


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

    def _project(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        attention_mask = (
            causal.view(1, 1, seq_len, seq_len)
            & valid_keys.view(batch, 1, 1, seq_len)
        )
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
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
            raise ValueError(
                f"Input length {seq_len} exceeds model limit {self.max_seq_len}."
            )
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
    ) -> List[int]:
        if not prompt_ids:
            raise ValueError("The prompt must contain at least one token.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if not 1 <= top_k <= self.output.out_features:
            raise ValueError(
                f"top_k must be between 1 and {self.output.out_features}."
            )
        if len(prompt_ids) > self.max_seq_len:
            raise ValueError("The prompt is longer than MINIBPE_SEQ_LEN.")

        self.eval()
        device = self.token_embedding.weight.device
        caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            None for _ in self.blocks
        ]
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

        while len(generated) < self.max_seq_len:
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
    mask = (
        torch.arange(seq_len, device=logits.device).unsqueeze(0)
        < valid_lens.unsqueeze(1)
    )
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


def train_model(
    model: MiniGPT,
    train_data: EncodedDataset,
    validation_data: EncodedDataset,
    config: Config,
    device: torch.device,
) -> Tuple[List[float], List[float]]:
    optimizer_kwargs = {"lr": config.learning_rate}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

    warmup_steps = max(1, config.max_steps * 2 // 100)

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (1 / 1000) + (1 - 1 / 1000) * step / warmup_steps
        progress = (step - warmup_steps) / max(1, config.max_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator().manual_seed(config.seed)
    training_losses: List[float] = []
    validation_losses: List[float] = []

    forward_model = model
    if config.compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("MINIBPE_COMPILE=1 requires torch.compile support.")
        forward_model = torch.compile(model)

    model.train()
    for step in range(config.max_steps):
        inputs, targets, valid_lens = sample_batch(
            train_data,
            config.batch_size,
            generator,
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
            step == 0
            or (step + 1) % config.eval_interval == 0
            or step + 1 == config.max_steps
        )
        if should_evaluate:
            validation_loss = estimate_loss(
                forward_model,
                validation_data,
                config,
                generator,
                device,
            )
            validation_losses.append(validation_loss)
            print(
                f"{step + 1:7d}/{config.max_steps}: "
                f"train_loss={training_losses[-1]:.4f} "
                f"val_loss={validation_loss:.4f}"
            )

    return training_losses, validation_losses


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


def main() -> None:
    config = Config.from_env()
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device()
    print(f"Device: {device}")

    train_samples, validation_samples = load_data(config.data_file)
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
    train_model(model, train_data, validation_data, config, device)


if __name__ == "__main__":
    main()
