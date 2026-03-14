"""
ONT k-mer pore model loader and utilities.

Loads a TSV file mapping each 6-mer to an expected current level (pA),
and provides conversion utilities for Bonito's base encoding to k-mer indices.
"""
import numpy as np
from pathlib import Path

BASES = "ACGT"
KMER_LEN = 6
NUM_KMERS = 4 ** KMER_LEN  # 4096

_BASE_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def kmer_to_index(kmer):
    """Convert a k-mer string (e.g. 'ACGTAC') to an integer index 0..4095."""
    idx = 0
    for base in kmer:
        idx = idx * 4 + _BASE_TO_INT[base]
    return idx


def index_to_kmer(idx, k=KMER_LEN):
    """Convert an integer index back to a k-mer string."""
    bases = []
    for _ in range(k):
        bases.append(BASES[idx % 4])
        idx //= 4
    return ''.join(reversed(bases))


class KmerModel:
    """
    Loads an ONT k-mer pore model from a TSV file.

    Expected TSV columns: kmer, level_mean, level_stdv, [optional others...]
    Lines starting with '#' are skipped.

    After loading, provides:
      - level_mean: np.ndarray [4096] of expected pA values per 6-mer
      - level_stdv: np.ndarray [4096] of standard deviations
      - global_mean, global_std: z-score normalization parameters
    """

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.level_mean = np.zeros(NUM_KMERS, dtype=np.float32)
        self.level_stdv = np.zeros(NUM_KMERS, dtype=np.float32)
        self._load()
        # Compute z-score normalization parameters across all k-mers
        self.global_mean = float(self.level_mean.mean())
        self.global_std = float(self.level_mean.std())

    def _load(self):
        """Parse TSV file, skipping comment/header lines."""
        with open(self.model_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('kmer'):
                    continue
                parts = line.split('\t')
                if len(parts) < 3:
                    continue
                kmer = parts[0]
                if len(kmer) != KMER_LEN:
                    continue
                idx = kmer_to_index(kmer)
                self.level_mean[idx] = float(parts[1])
                self.level_stdv[idx] = float(parts[2])

    def bases_to_kmer_ids(self, bases):
        """
        Convert an array of Bonito base indices (A=1,C=2,G=3,T=4) to 6-mer indices.

        Args:
            bases: np.ndarray [L] of uint8 with values 1-4 (0=N/pad)

        Returns:
            np.ndarray [L - KMER_LEN + 1] of int32, k-mer indices 0..4095
        """
        # Map from Bonito encoding (1-indexed) to 0-indexed
        zero_indexed = np.clip(bases.astype(np.int32) - 1, 0, 3)
        n_kmers = len(zero_indexed) - KMER_LEN + 1
        if n_kmers <= 0:
            return np.array([], dtype=np.int32)
        kmer_ids = np.zeros(n_kmers, dtype=np.int32)
        for i in range(n_kmers):
            idx = 0
            for j in range(KMER_LEN):
                idx = idx * 4 + zero_indexed[i + j]
            kmer_ids[i] = idx
        return kmer_ids

    def get_expected_signal(self, kmer_ids):
        """
        Look up expected signal levels for k-mer indices, z-score normalized.

        Args:
            kmer_ids: np.ndarray of int32 k-mer indices

        Returns:
            np.ndarray of float32, z-score normalized expected signal values
        """
        raw = self.level_mean[kmer_ids]
        return (raw - self.global_mean) / self.global_std
