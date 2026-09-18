from __future__ import annotations

import math


def expected_random_ap(total: int, relevant: int) -> float:
    """Return the exact expected AP of a uniform random permutation."""
    if not 0 < relevant <= total:
        raise ValueError("relevant must be between 1 and total")
    harmonic = sum(1.0 / rank for rank in range(1, total + 1))
    return harmonic / total + (relevant - 1) * (total - harmonic) / (
        total * (total - 1)
    )


def probability_perfect_topk(total: int, relevant: int, k: int) -> float:
    """Return P(Precision@K = 1) under sampling without replacement."""
    if not 0 <= relevant <= total:
        raise ValueError("relevant must be between 0 and total")
    if k < 0:
        raise ValueError("k must be non-negative")
    if k > relevant or k > total:
        return 0.0
    return math.comb(relevant, k) / math.comb(total, k)
