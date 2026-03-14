"""
Multi-head cross-attention module for RawHash-guided basecalling.

Q comes from signal representation, K/V come from reference embeddings.
Uses F.scaled_dot_product_attention for Flash Attention support.
No rotary embeddings (cross-sequence), no sliding window (full attention over reference).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from bonito.nn import Module, register


@register
class CrossAttention(Module):

    def __init__(self, d_model, nhead, bias=False):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.Wq = nn.Linear(d_model, d_model, bias=bias)
        self.Wkv = nn.Linear(d_model, 2 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x, R):
        """
        Args:
            x: [B, T, d_model] signal query embeddings
            R: [B, L, d_model] reference key/value embeddings

        Returns:
            [B, T, d_model] cross-attended output
        """
        B, T, _ = x.shape
        L = R.shape[1]

        q = self.Wq(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        kv = self.Wkv(R).view(B, L, 2, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # each [B, H, L, D]

        attn_out = F.scaled_dot_product_attention(q, k, v)  # [B, H, T, D]
        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(attn_out)

    def to_dict(self, include_weights=False):
        if include_weights:
            raise NotImplementedError
        return {
            'd_model': self.d_model,
            'nhead': self.nhead,
        }
