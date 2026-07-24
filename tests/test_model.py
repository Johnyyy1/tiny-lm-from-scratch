import pytest
import torch

from tiny_lm.model import MiniGPT
from tiny_lm.training import initialize_runtime


def test_model_logits_shape(tiny_config) -> None:
    model = MiniGPT(tiny_config(), vocab_size=8).eval()
    token_ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
    logits = model(token_ids, torch.tensor([3, 3]))
    assert logits.shape == (2, 3, 8)


def test_model_cannot_see_future_tokens(tiny_config) -> None:
    model = MiniGPT(tiny_config(), vocab_size=8).eval()
    first = torch.tensor([[1, 2, 3, 4]])
    second = torch.tensor([[1, 2, 6, 7]])
    valid_lens = torch.tensor([4])
    first_logits = model(first, valid_lens)
    second_logits = model(second, valid_lens)
    torch.testing.assert_close(first_logits[:, :2], second_logits[:, :2])


def test_tied_embeddings_share_weights(tiny_config) -> None:
    model = MiniGPT(tiny_config(), vocab_size=8)
    assert model.output.weight is model.token_embedding.weight
    assert model.output.weight.data_ptr() == model.token_embedding.weight.data_ptr()


def test_same_seed_initializes_same_model(tiny_config) -> None:
    initialize_runtime(123, "cpu")
    first = MiniGPT(tiny_config(), vocab_size=8)
    initialize_runtime(123, "cpu")
    second = MiniGPT(tiny_config(), vocab_size=8)
    for first_parameter, second_parameter in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(first_parameter, second_parameter)


def test_invalid_head_count_is_rejected(tiny_config) -> None:
    with pytest.raises(ValueError, match="must divide"):
        tiny_config(d_model=10, num_heads=3).validate()
