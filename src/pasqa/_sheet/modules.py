#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Modules for SSLMOS: Projection, MoraCrossAttention, RoPE, make_non_pad_mask."""

from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Projection (from sheet/modules/ldnet/modules.py)
# ---------------------------------------------------------------------------

class Projection(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        activation,
        output_type,
        _output_dim,
        output_step=1.0,
        range_clipping=False,
    ):
        super(Projection, self).__init__()
        self.output_type = output_type
        self.range_clipping = range_clipping
        if output_type == "scalar":
            output_dim = 1
            if range_clipping:
                self.proj = nn.Tanh()
        elif output_type == "categorical":
            output_dim = _output_dim
            self.output_step = output_step
        else:
            raise NotImplementedError("wrong output_type: {}".format(output_type))

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, inference=False):
        output = self.net(x)

        if self.output_type == "scalar":
            if self.range_clipping:
                return self.proj(output) * 2.0 + 3
            else:
                return output
        else:
            if inference:
                return torch.argmax(output, dim=-1) * self.output_step + 1
            else:
                return output


# ---------------------------------------------------------------------------
# make_non_pad_mask (from sheet/modules/utils.py)
# ---------------------------------------------------------------------------

def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.long().tolist()

    bs = int(len(lengths))
    if maxlen is None:
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)
    else:
        assert xs is None
        assert maxlen >= int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    return ~make_pad_mask(lengths, xs, length_dim)


# ---------------------------------------------------------------------------
# RotaryPositionalEmbedding and MoraCrossAttention
# (from sheet/modules/mora_attention.py)
# ---------------------------------------------------------------------------

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) implementation using PyTorch only."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(
                f"RotaryPositionalEmbedding requires even dim, got {dim}"
            )
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._seq_len_cached = 0
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq.to(dtype))
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        assert dim == self.dim, f"Expected dim={self.dim}, got {dim}"

        self._update_cache(seq_len, x.device, x.dtype)

        assert self._cos_cached is not None and self._sin_cached is not None
        cos = self._cos_cached[:seq_len]
        sin = self._sin_cached[:seq_len]

        return (x * cos) + (self._rotate_half(x) * sin)


class MoraCrossAttention(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        mora_vocab_size: int,
        mora_emb_dim: int = 256,
        mora_transformer_layers: int = 1,
        mora_transformer_heads: int = 4,
        mora_ffn_dim: int = 512,
        mora_dropout: float = 0.1,
        mora_max_len: int = 128,
        attn_dim: int = 256,
        attn_heads: int = 4,
        attn_dropout: float = 0.1,
        attn_alpha_init: float = 0.1,
        mora_pos_encoding: str = "rope",
    ) -> None:
        super().__init__()

        if mora_pos_encoding not in ["learned", "rope"]:
            raise ValueError(
                f"mora_pos_encoding must be 'learned' or 'rope', got '{mora_pos_encoding}'"
            )

        self.mora_pos_encoding = mora_pos_encoding
        self.mora_max_len = int(mora_max_len)
        self.mora_embedding = nn.Embedding(
            mora_vocab_size, mora_emb_dim, padding_idx=0
        )

        if self.mora_pos_encoding == "learned":
            self.mora_pos_embedding: Optional[nn.Embedding] = nn.Embedding(
                self.mora_max_len, mora_emb_dim
            )
            self.mora_rope: Optional[RotaryPositionalEmbedding] = None
        else:
            self.mora_pos_embedding = None
            self.mora_rope = RotaryPositionalEmbedding(
                dim=mora_emb_dim, max_seq_len=mora_max_len * 2
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=mora_emb_dim,
            nhead=mora_transformer_heads,
            dim_feedforward=mora_ffn_dim,
            dropout=mora_dropout,
            batch_first=True,
        )
        self.mora_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=mora_transformer_layers
        )
        self.mora_q_proj = nn.Linear(encoder_dim, attn_dim)
        self.mora_k_proj = nn.Linear(mora_emb_dim, attn_dim)
        self.mora_v_proj = nn.Linear(mora_emb_dim, attn_dim)
        self.mora_attn = nn.MultiheadAttention(
            attn_dim, attn_heads, dropout=attn_dropout, batch_first=True
        )
        self.mora_out_proj = nn.Linear(attn_dim, encoder_dim)
        self.mora_layer_norm = nn.LayerNorm(encoder_dim)
        self.mora_alpha = nn.Parameter(torch.tensor(attn_alpha_init))

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        mora_idxs: torch.Tensor,
        mora_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mora_idxs is None:
            raise ValueError("mora_idxs must be provided when use_mora=True.")
        if mora_lengths is None:
            mora_lengths = (mora_idxs != 0).sum(dim=1)
        mora_lengths = mora_lengths.to(mora_idxs.device)

        batch, mora_len = mora_idxs.size(0), mora_idxs.size(1)

        if self.mora_pos_encoding == "learned":
            if mora_len > self.mora_max_len:
                raise ValueError(
                    f"mora length {mora_len} exceeds mora_max_len={self.mora_max_len}"
                )
            pos = torch.arange(mora_len, device=mora_idxs.device).unsqueeze(0).expand(
                batch, -1
            )
            assert self.mora_pos_embedding is not None
            mora_emb = self.mora_embedding(mora_idxs) + self.mora_pos_embedding(pos)
        else:
            assert self.mora_rope is not None
            mora_emb = self.mora_embedding(mora_idxs)
            mora_emb = self.mora_rope(mora_emb)

        mora_pad_mask = torch.arange(mora_len, device=mora_idxs.device).unsqueeze(
            0
        ).expand(batch, -1) >= mora_lengths.unsqueeze(1)
        mora_enc = self.mora_encoder(mora_emb, src_key_padding_mask=mora_pad_mask)

        q = self.mora_q_proj(encoder_outputs)
        k = self.mora_k_proj(mora_enc)
        v = self.mora_v_proj(mora_enc)
        attn_out, _ = self.mora_attn(
            q, k, v, key_padding_mask=mora_pad_mask, need_weights=False
        )
        attn_out = self.mora_out_proj(attn_out)
        return self.mora_layer_norm(encoder_outputs + self.mora_alpha * attn_out)
