"""Ranking metrics and query-level bootstrap utilities."""

import numpy as np


def reciprocal_rank_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive.")
    for rank, product_id in enumerate(retrieved[:k], start=1):
        if product_id in relevant:
            return 1.0 / rank
    return 0.0


def bootstrap_mean_ci(
    values: list[float],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("Bootstrap CI requires a non-empty 1D score array.")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, scores.size, size=(n_bootstrap, scores.size))
    means = scores[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "mean": float(scores.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "n_queries": int(scores.size),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }


def paired_bootstrap_difference(
    scores_a: list[float],
    scores_b: list[float],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.size == 0 or a.size != b.size:
        raise ValueError("Paired bootstrap requires equal, non-empty 1D score arrays.")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")

    differences = a - b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(n_bootstrap, differences.size))
    means = differences[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "mean_difference": float(differences.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "n_queries": int(differences.size),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }
