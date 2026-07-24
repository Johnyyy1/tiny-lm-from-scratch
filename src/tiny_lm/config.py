"""Configuration values shared by training and inference."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MIN_VOCAB_SIZE = 257


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc


@dataclass(frozen=True)
class Config:
    """Complete model and training configuration."""

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
        """Build a configuration from ``MINIBPE_*`` environment variables."""

        config = cls(
            data_file=Path(os.environ.get("MINIBPE_DATA_FILE", "data/input.txt")),
            vocab_size=_env_int("MINIBPE_VOCAB_SIZE", 5000),
            seq_len=_env_int("MINIBPE_SEQ_LEN", 512),
            d_model=_env_int("MINIBPE_D_MODEL", 1024),
            num_heads=_env_int("MINIBPE_NUM_HEADS", 8),
            d_ff=_env_int("MINIBPE_D_FF", 2048),
            num_layers=_env_int("MINIBPE_NUM_LAYERS", 2),
            batch_size=_env_int("MINIBPE_BATCH_SIZE", 16),
            max_steps=_env_int("MINIBPE_MAX_STEPS", 200000),
            eval_interval=_env_int("MINIBPE_EVAL_INTERVAL", 100),
            eval_batches=_env_int("MINIBPE_EVAL_BATCHES", 4),
            learning_rate=_env_float("MINIBPE_LEARNING_RATE", 3e-4),
            dropout=_env_float("MINIBPE_DROPOUT", 0.1),
            max_token_length=_env_int("MINIBPE_MAX_TOKEN_LENGTH", 40),
            compile_model=os.environ.get("MINIBPE_COMPILE", "0") == "1",
            seed=_env_int("MINIBPE_SEED", 42),
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
        if self.vocab_size < MIN_VOCAB_SIZE:
            raise ValueError(
                "MINIBPE_VOCAB_SIZE must be at least 257 "
                "(256 byte tokens plus the end-of-text token)."
            )
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
        parsed = dict(values)
        parsed["data_file"] = Path(parsed["data_file"])
        config = cls(**parsed)
        config.validate()
        return config
