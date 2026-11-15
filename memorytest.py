# MemoryTest - Correctness tests for associative memory as a library
# Part of the Kapildhinwa2010-dev Toolkit.

"""WaveMemory - Hebbian superposition memory with interference recall.

Lineage: reserv.dean-part-2 (helix.py wave arena). Patterns are written as
superposed outer products into a Hebbian lattice. Two recall modes:

- recall(): exact nearest stored pattern by correlation (needs stored
  copies; this is the mode that actually worked in our experiments).
- recall_from_lattice(): Hopfield-style iterative sign updates using ONLY
  the accumulated lattice, so it works after forget_copies().

HONEST LIMITATIONS (measured, see README): superposition capacity is
roughly 0.1-0.14 * dim uncorrelated patterns; recall degrades sharply
under noise once patterns correlate, and the lattice mode is NOT safe
under heavy noise in our experiments.
"""

from __future__ import annotations

import numpy as np

__all__ = ["WaveMemory", "hamming_distance", "noise_pattern"]


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("patterns must have the same shape")
    return int((a != b).sum())


def noise_pattern(shape, flip_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Random +/-1 pattern."""
    if not 0.0 <= flip_prob <= 1.0:
        raise ValueError("flip_prob must be in [0, 1]")
    return rng.choice(np.array([-1, 1]), size=shape)


class WaveMemory:
    """Hebbian superposition memory with interference-based recall."""

    def __init__(self, dim: int) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = int(dim)
        self.lattice = np.zeros((self.dim, self.dim), dtype=np.float64)
        self.patterns: list = []

    def write(self, pattern: np.ndarray) -> None:
        pattern = np.asarray(pattern, dtype=np.float64).ravel()
        if pattern.size != self.dim:
            raise ValueError("pattern size must equal dim")
        if not np.all(np.isin(pattern, (-1.0, 1.0))):
            raise ValueError("patterns must contain only -1 and +1")
        self.lattice += np.outer(pattern, pattern)
        self.patterns.append(pattern.copy())

    def write_many(self, patterns) -> None:
        for pattern in patterns:
            self.write(pattern)

    def recall(self, cue: np.ndarray) -> np.ndarray:
        """Return the stored pattern most correlated with ``cue``.

        Uses the stored copies, so it is exact but needs them kept.
        """
        cue = np.asarray(cue, dtype=np.float64).ravel()
        if cue.size != self.dim:
            raise ValueError("cue size must equal dim")
        if not self.patterns:
            raise ValueError("memory is empty")
        scores = [float(cue @ p) for p in self.patterns]
        return self.patterns[int(np.argmax(scores))].copy()

    def recall_from_lattice(self, cue: np.ndarray, steps: int = 3) -> np.ndarray:
        """Hopfield-style recall using ONLY the Hebbian lattice.

        Iteratively applies sign(lattice @ state) starting from the cue.
        Works after forget_copies(); degrades when stored patterns
        correlate or the cue is heavily corrupted.
        """
        cue = np.asarray(cue, dtype=np.float64).ravel()
        if cue.size != self.dim:
            raise ValueError("cue size must equal dim")
        if not self.patterns and not self.lattice.any():
            raise ValueError("memory is empty")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        state = np.where(cue >= 0, 1.0, -1.0)
        for _ in range(int(steps)):
            state = np.where(self.lattice @ state >= 0, 1.0, -1.0)
        return state

    def forget_copies(self) -> None:
        """Drop stored copies; only the Hebbian lattice remains."""
        self.patterns = []

    def cleanup(self, noisy: np.ndarray, max_flips: int = 8) -> np.ndarray:
        """Recall and verify; raise if the result is too far from the cue."""
        recovered = self.recall(noisy)
        flips = hamming_distance(recovered, np.sign(noisy))
        if flips > max_flips * self.dim // 100:
            raise ValueError("cleanup failed: {} flips".format(flips))
        return recovered

    def __len__(self) -> int:
        return len(self.patterns)