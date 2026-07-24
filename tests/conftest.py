from pathlib import Path

import pytest

from tiny_lm.config import Config
from tiny_lm.tokenizer import BPETokenizer


def make_tiny_config(**overrides) -> Config:
    values = {
        "data_file": Path("data/input.txt"),
        "vocab_size": 257,
        "seq_len": 8,
        "d_model": 12,
        "num_heads": 3,
        "d_ff": 24,
        "num_layers": 2,
        "batch_size": 2,
        "max_steps": 4,
        "eval_interval": 2,
        "eval_batches": 2,
        "learning_rate": 3e-4,
        "dropout": 0.0,
        "max_token_length": 8,
        "compile_model": False,
        "seed": 42,
    }
    values.update(overrides)
    return Config(**values)


@pytest.fixture
def tiny_config():
    return make_tiny_config


@pytest.fixture
def tokenizer() -> BPETokenizer:
    return BPETokenizer.train(["Hello world", "Hello there"], 270, 8)
