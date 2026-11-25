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


def rank_metrics(ranks, k: int = 10) -> dict[str, np.ndarray]:
    """Chỉ số theo từng query từ hạng 0-based của đáp án (mỗi query đúng một đáp án).

    MRR@k và NDCG@k là bản giới hạn trong top-k: đáp án ngoài top-k tính 0.
    Hạng được tính theo kiểu bi quan (điểm hòa tính là thua), xem evaluate.rank_queries.
    """
    r = np.asarray(ranks, dtype=np.int64)
    in_k = r < k
    out = {f"Recall@{n}": (r < n).astype(np.float64) for n in (1, 5, 10)}
    out[f"MRR@{k}"] = np.where(in_k, 1.0 / (r + 1), 0.0)
    out[f"NDCG@{k}"] = np.where(in_k, 1.0 / np.log2(r + 2), 0.0)
    return out


def cluster_bootstrap_mean_ci(
    values,
    clusters,
    *,
    n_bootstrap: int = 2_000,
    seed: int = 42,
    chunk: int = 500,
) -> dict[str, float]:
    """Bootstrap theo CỤM (vd. group_id của các sản phẩm gần trùng ảnh).

    Lấy mẫu có hoàn lại các cụm thay vì từng query, vì query trong cùng một cụm không độc lập;
    lấy mẫu theo query sẽ cho khoảng tin cậy hẹp hơn thực tế. Cùng seed + cùng cụm
    => cùng các lần lấy mẫu cho mọi chỉ số.
    """
    v = np.asarray(values, dtype=np.float64)
    c = np.asarray(clusters)
    if v.ndim != 1 or v.size == 0 or c.shape != v.shape:
        raise ValueError("Cần mảng 1D không rỗng, values và clusters cùng kích thước.")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")
    uniq, inv = np.unique(c, return_inverse=True)
    n_cl = uniq.size
    sums = np.bincount(inv, weights=v, minlength=n_cl)
    counts = np.bincount(inv, minlength=n_cl).astype(np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for s in range(0, n_bootstrap, chunk):
        m = min(chunk, n_bootstrap - s)
        idx = rng.integers(0, n_cl, size=(m, n_cl))
        means[s:s + m] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "mean": float(v.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "n_queries": int(v.size),
        "n_clusters": int(n_cl),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }