"""加权分布工具：分位数、直方图、与地球分布的距离（Wasserstein、KS）、幂律尾部拟合。"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, qs) -> np.ndarray:
    """加权分位数（qs 取值 0–1）。用于面积加权的高程、坡度、距海距离分布。"""
    v = np.asarray(values, float).ravel()
    w = np.asarray(weights, float).ravel()
    keep = np.isfinite(v) & (w > 0)
    v, w = v[keep], w[keep]
    if v.size == 0:
        return np.full(len(np.atleast_1d(qs)), np.nan)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    cw = (cw - 0.5 * w) / cw[-1]
    return np.interp(np.atleast_1d(qs), cw, v)


def weighted_histogram(values, weights, bins) -> dict:
    v = np.asarray(values, float).ravel()
    w = np.asarray(weights, float).ravel()
    keep = np.isfinite(v)
    h, edges = np.histogram(v[keep], bins=bins, weights=w[keep])
    total = h.sum()
    return {"edges": edges.tolist(), "fraction": (h / total if total > 0 else h).tolist()}


def wasserstein(values_a, weights_a, values_b, weights_b) -> float:
    """两个加权分布之间的一阶 Wasserstein 距离（单位与数值相同，例如米）。"""
    a, wa = np.asarray(values_a, float).ravel(), np.asarray(weights_a, float).ravel()
    b, wb = np.asarray(values_b, float).ravel(), np.asarray(weights_b, float).ravel()
    ka, kb = np.isfinite(a) & (wa > 0), np.isfinite(b) & (wb > 0)
    return float(wasserstein_distance(a[ka], b[kb], wa[ka], wb[kb]))


def weighted_ks(values_a, weights_a, values_b, weights_b) -> float:
    """加权 KS 统计量：两个累积分布的最大差。"""
    def cdf(v, w, grid):
        o = np.argsort(v)
        v, w = v[o], w[o]
        cw = np.cumsum(w) / w.sum()
        return np.interp(grid, v, cw, left=0.0, right=1.0)

    a, wa = np.asarray(values_a, float).ravel(), np.asarray(weights_a, float).ravel()
    b, wb = np.asarray(values_b, float).ravel(), np.asarray(weights_b, float).ravel()
    ka, kb = np.isfinite(a) & (wa > 0), np.isfinite(b) & (wb > 0)
    a, wa, b, wb = a[ka], wa[ka], b[kb], wb[kb]
    grid = np.union1d(a, b)
    return float(np.abs(cdf(a, wa, grid) - cdf(b, wb, grid)).max())


def powerlaw_ccdf_slope(sizes: np.ndarray, s_min: float, s_max: float | None = None, n_thresholds: int = 12) -> dict:
    """累积分布 N(≥s) ∝ s^(−b) 的拟合（Korčak 经验律）。在 [s_min, s_max] 内取对数等距阈值，
    对 log N 与 log s 做最小二乘，返回 b（取正值）与参与拟合的样本数。"""
    s = np.asarray(sizes, float)
    s = s[s >= s_min]
    if s_max is None:
        s_max = s.max() if s.size else s_min
    if s.size < 5 or s_max <= s_min * 2:
        return {"b": float("nan"), "n": int(s.size), "s_min": s_min, "s_max": s_max}
    th = np.geomspace(s_min, s_max, n_thresholds)
    n = np.array([(s >= t).sum() for t in th], float)
    ok = n > 0
    slope = np.polyfit(np.log(th[ok]), np.log(n[ok]), 1)[0]
    return {"b": float(-slope), "n": int(s.size), "s_min": float(s_min), "s_max": float(s_max)}
