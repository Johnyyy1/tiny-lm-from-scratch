"""Causal self-attention and Transformer blocks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

KVCache = tuple[torch.Tensor, torch.Tensor]


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
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

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
        cache: KVCache | None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Process one token and append its keys and values to the cache."""

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
        x = x + self.dropout(self.attention(self.norm1(x), valid_lens))
        return x + self.dropout(self.ffn(self.norm2(x)))

    def step(
        self,
        x: torch.Tensor,
        cache: KVCache | None,
    ) -> tuple[torch.Tensor, KVCache]:
        attention, cache = self.attention.step(self.norm1(x), cache)
        x = x + self.dropout(attention)
        return x + self.dropout(self.ffn(self.norm2(x))), cache
