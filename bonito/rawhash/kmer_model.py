"""
ONT k-mer pore model loader and utilities.

Supports both legacy 6-mer TSV files (tab-separated with header)
and newer 9-mer level files (space-separated, no header).
Auto-detects k-mer length from the first entry.
"""
import numpy as np
from pathlib import Path

BASES = "ACGT"

_BASE_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def kmer_to_index(kmer):
    """Convert a k-mer string (e.g. 'ACGTAC') to an integer index."""
    idx = 0
    for base in kmer:
        idx = idx * 4 + _BASE_TO_INT[base]
    return idx


def index_to_kmer(idx, k):
    """Convert an integer index back to a k-mer string."""
    bases = []
    for _ in range(k):
        bases.append(BASES[idx % 4])
        idx //= 4
    return ''.join(reversed(bases))


class KmerModel:
    """
    Loads an ONT k-mer pore model from a file.

    Supports two formats:
      1) Legacy TSV: tab-separated, columns kmer/level_mean/level_stdv, optional header
      2) 9-mer levels: space-separated, columns kmer/level, no header

    Auto-detects k-mer length and number of k-mers from the data.
    """

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.kmer_len = None
        self.num_kmers = None
        self.level_mean = None
        self._load()
        self.global_mean = float(self.level_mean.mean())
        self.global_std = float(self.level_mean.std())
        if self.global_std == 0:
            self.global_std = 1.0

    def _load(self):
        """Parse model file, auto-detecting format."""
        entries = []
        with open(self.model_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('kmer'):
                    continue
                # Try tab-separated first, then space-separated
                parts = line.split('\t') if '\t' in line else line.split()
                if len(parts) < 2:
                    continue
                kmer = parts[0]
                if not all(c in 'ACGT' for c in kmer):
                    continue
                level = float(parts[1])
                entries.append((kmer, level))

        if not entries:
            raise ValueError(f"No valid k-mer entries found in {self.model_path}")

        self.kmer_len = len(entries[0][0])
        self.num_kmers = 4 ** self.kmer_len
        self.level_mean = np.zeros(self.num_kmers, dtype=np.float32)

        for kmer, level in entries:
            if len(kmer) != self.kmer_len:
                continue
            idx = kmer_to_index(kmer)
            self.level_mean[idx] = level

    def bases_to_kmer_ids(self, bases):
        """
        Convert an array of Bonito base indices (A=1,C=2,G=3,T=4) to k-mer indices.

        Args:
            bases: np.ndarray [L] of uint8 with values 1-4 (0=N/pad)

        Returns:
            np.ndarray [L - kmer_len + 1] of int32, k-mer indices
        """
        zero_indexed = np.clip(bases.astype(np.int32) - 1, 0, 3)
        n_kmers = len(zero_indexed) - self.kmer_len + 1
        if n_kmers <= 0:
            return np.array([], dtype=np.int32)
        kmer_ids = np.zeros(n_kmers, dtype=np.int32)
        for i in range(n_kmers):
            idx = 0
            for j in range(self.kmer_len):
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
