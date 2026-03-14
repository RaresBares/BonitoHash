"""
Reference signal embedder for cross-attention.

Converts reference k-mer indices and expected signal values into
d_model-dimensional embeddings suitable for cross-attention K/V.

For large k-mer vocabularies (e.g. 9-mer = 262144 entries), uses
a compositional embedding: the k-mer index is split into high and low
parts, each embedded separately and summed. This keeps the embedding
table small while preserving expressivity.
"""
import torch
import torch.nn as nn

from bonito.nn import Module, register


@register
class ReferenceEmbedder(Module):
    """
    Two embedding pathways:
      1. K-mer identity embedding (compositional for large vocabularies)
      2. Expected signal projection: Linear(1, d_model//2)

    Combined via concatenation, linear projection, and RMSNorm.
    Output: [B, L, d_model]
    """

    def __init__(self, d_model, num_kmers=4096):
        super().__init__()
        self.d_model = d_model
        self.num_kmers = num_kmers
        half_d = d_model // 2

        # For large vocabularies (>16384), use compositional embedding
        # Split k-mer index into high and low parts
        if num_kmers > 16384:
            # sqrt decomposition: e.g. 262144 = 512 * 512
            self.embed_size = int(num_kmers ** 0.5)
            if self.embed_size * self.embed_size < num_kmers:
                self.embed_size += 1
            self.kmer_embed_high = nn.Embedding(self.embed_size, half_d)
            self.kmer_embed_low = nn.Embedding(self.embed_size, half_d)
            self.compositional = True
        else:
            self.kmer_embed = nn.Embedding(num_kmers, half_d)
            self.compositional = False

        self.signal_proj = nn.Linear(1, half_d)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, kmer_ids, expected_signals):
        """
        Args:
            kmer_ids: [B, L] int tensor of k-mer indices
            expected_signals: [B, L] float tensor of z-score normalized expected signals

        Returns:
            R: [B, L, d_model] reference embeddings
        """
        if self.compositional:
            high = kmer_ids // self.embed_size
            low = kmer_ids % self.embed_size
            kmer_emb = self.kmer_embed_high(high) + self.kmer_embed_low(low)
        else:
            kmer_emb = self.kmer_embed(kmer_ids)

        sig_emb = self.signal_proj(expected_signals.unsqueeze(-1))
        combined = torch.cat([kmer_emb, sig_emb], dim=-1)
        return self.norm(self.out_proj(combined))

    def to_dict(self, include_weights=False):
        if include_weights:
            raise NotImplementedError
        return {
            'd_model': self.d_model,
            'num_kmers': self.num_kmers,
        }
