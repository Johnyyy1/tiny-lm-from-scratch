"""Autoregressive text generation using a per-layer KV cache."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from tiny_lm.attention import KVCache
from tiny_lm.model import MiniGPT


def _step(
    model: MiniGPT,
    token_id: int,
    position: int,
    caches: list[KVCache | None],
) -> tuple[torch.Tensor, list[KVCache]]:
    device = model.token_embedding.weight.device
    token = torch.tensor([[token_id]], device=device)
    position_id = torch.tensor([position], device=device)
    hidden = model.token_embedding(token) + model.position_embedding(position_id)
    new_caches = []
    for block, cache in zip(model.blocks, caches):
        hidden, cache = block.step(hidden, cache)
        new_caches.append(cache)
    return model.output(model.final_norm(hidden))[0, -1], new_caches


@torch.inference_mode()
def generate(
    model: MiniGPT,
    prompt_ids: Sequence[int],
    stop_id: int,
    temperature: float,
    top_k: int,
    max_new_tokens: int | None = None,
    stop_on_eot: bool = True,
) -> list[int]:
    """Generate token IDs autoregressively from a non-empty prompt."""

    if not prompt_ids:
        raise ValueError("The prompt must contain at least one token.")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive.")
    if not 1 <= top_k <= model.output.out_features:
        raise ValueError(f"top_k must be between 1 and {model.output.out_features}.")
    if len(prompt_ids) > model.max_seq_len:
        raise ValueError("The prompt is longer than MINIBPE_SEQ_LEN.")
    if max_new_tokens is not None and max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive.")

    model.eval()
    caches: list[KVCache | None] = [None for _ in model.blocks]
    generated = list(prompt_ids)
    logits = None
    for position, token_id in enumerate(prompt_ids):
        logits, caches = _step(model, token_id, position, caches)

    generation_limit = model.max_seq_len
    if max_new_tokens is not None:
        generation_limit = min(generation_limit, len(prompt_ids) + max_new_tokens)

    while len(generated) < generation_limit:
        top_logits, top_indices = torch.topk(logits, top_k)
        probabilities = F.softmax(top_logits / temperature, dim=-1)
        sampled = torch.multinomial(probabilities, 1).item()
        token_id = top_indices[sampled].item()
        if token_id == stop_id and stop_on_eot:
            break
        generated.append(token_id)
        logits, caches = _step(model, token_id, len(generated) - 1, caches)
    return generated
