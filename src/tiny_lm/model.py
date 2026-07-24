"""Decoder-only Transformer language model."""

from __future__ import annotations

import torch
from torch import nn

from tiny_lm.attention import TransformerBlock
from tiny_lm.config import Config


class MiniGPT(nn.Module):
    """A small GPT-style Transformer with learned positional embeddings."""

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
