"""
Reference signal embedder for cross-attention.

Converts reference k-mer indices and expected signal values into
d_model-dimensional embeddings suitable for cross-attention K/V.
"""
import torch
import torch.nn as nn

from bonito.nn import Module, register


@register
class ReferenceEmbedder(Module):
    """
    Two embedding pathways:
      1. Learned k-mer embedding: Embedding(4096, d_model//2)
      2. Expected signal projection: Linear(1, d_model//2)

    Combined via concatenation, linear projection, and RMSNorm.
    Output: [B, L, d_model]
    """

    def __init__(self, d_model, num_kmers=4096):
        super().__init__()
        self.d_model = d_model
        self.num_kmers = num_kmers
        half_d = d_model // 2

        self.kmer_embed = nn.Embedding(num_kmers, half_d)
        self.signal_proj = nn.Linear(1, half_d)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, kmer_ids, expected_signals):
        """
        Args:
            kmer_ids: [B, L] int tensor of k-mer indices (0..4095)
            expected_signals: [B, L] float tensor of z-score normalized expected signals

        Returns:
            R: [B, L, d_model] reference embeddings
        """
        kmer_emb = self.kmer_embed(kmer_ids)                          # [B, L, d_model//2]
        sig_emb = self.signal_proj(expected_signals.unsqueeze(-1))     # [B, L, d_model//2]
        combined = torch.cat([kmer_emb, sig_emb], dim=-1)             # [B, L, d_model]
        return self.norm(self.out_proj(combined))

    def to_dict(self, include_weights=False):
        if include_weights:
            raise NotImplementedError
        return {
            'd_model': self.d_model,
            'num_kmers': self.num_kmers,
        }
