import pytest
import torch

from tiny_lm.generation import generate
from tiny_lm.model import MiniGPT


def test_cached_and_uncached_logits_match(tiny_config) -> None:
    model = MiniGPT(tiny_config(), vocab_size=8).eval()
    token_ids = torch.tensor([[1, 2, 3, 4]])
    expected = model(token_ids, torch.tensor([4]))

    caches = [None for _ in model.blocks]
    cached_logits = []
    for position in range(token_ids.shape[1]):
        token = token_ids[:, position : position + 1]
        position_id = torch.tensor([position])
        hidden = model.token_embedding(token) + model.position_embedding(position_id)
        new_caches = []
        for block, cache in zip(model.blocks, caches):
            hidden, cache = block.step(hidden, cache)
            new_caches.append(cache)
        caches = new_caches
        cached_logits.append(model.output(model.final_norm(hidden)))

    torch.testing.assert_close(torch.cat(cached_logits, dim=1), expected)


def test_generate_respects_token_limit(tiny_config) -> None:
    model = MiniGPT(tiny_config(seq_len=6), vocab_size=8)
    torch.manual_seed(7)
    result = generate(model, [1, 2], stop_id=0, temperature=1.0, top_k=7, max_new_tokens=2)
    assert 2 <= len(result) <= 4
    assert result[:2] == [1, 2]


def test_generate_can_continue_after_eot(tiny_config) -> None:
    model = MiniGPT(tiny_config(seq_len=4), vocab_size=2)
    with torch.no_grad():
        model.output.weight.zero_()
    result = generate(
        model,
        [1],
        stop_id=0,
        temperature=1.0,
        top_k=1,
        max_new_tokens=2,
        stop_on_eot=False,
    )
    assert result == [1, 0, 0]


def test_generate_rejects_empty_prompt(tiny_config) -> None:
    model = MiniGPT(tiny_config(), vocab_size=8)
    with pytest.raises(ValueError, match="non-empty|at least one"):
        generate(model, [], stop_id=0, temperature=1.0, top_k=4)
