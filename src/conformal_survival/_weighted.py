"""Weighted split-conformal quantile (Tibshirani et al., 2019)."""
from __future__ import annotations

import numpy as np


def weighted_quantile_with_inf(scores, weights, test_weights, level):
    """eta(x) = Quantile(level; sum_i p_i(x) delta_{V_i} + p_inf(x) delta_{+inf}).

    With p_i(x) = W_i / (sum_j W_j + w(x)) and p_inf(x) = w(x) / (sum_j W_j + w(x)),
    and Quantile(beta; Q) = sup{z : Q(Z <= z) < beta}, as in step 6 of Algorithm 1
    of Candes, Lei & Ren (2023). Vectorized over test points.

    Parameters
    ----------
    scores : (n,) calibration conformity scores V_i
    weights : (n,) calibration weights W_i >= 0
    test_weights : (m,) test weights w(x) >= 0 (may be +inf)
    level : the target level, 1 - alpha

    Returns
    -------
    (m,) array; +inf where the finite scores carry too little mass.
    """
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    test_weights = np.atleast_1d(np.asarray(test_weights, dtype=float))
    if scores.shape != weights.shape:
        raise ValueError("scores and weights must have the same shape")
    if np.any(weights < 0) or np.any(test_weights < 0):
        raise ValueError("weights must be non-negative")

    eta = np.full(test_weights.shape, np.inf)
    if scores.size == 0:
        return eta
    order = np.argsort(scores, kind="stable")
    v = scores[order]
    cum = np.cumsum(weights[order])
    # The CDF of the weighted distribution at v_(k) is cum[k] / (cum[-1] + w(x));
    # eta is the first v_(k) where that reaches `level`.
    with np.errstate(invalid="ignore"):
        threshold = level * (cum[-1] + test_weights)
    k = np.searchsorted(cum, threshold, side="left")
    ok = np.isfinite(threshold) & (k < v.size)
    eta[ok] = v[k[ok]]
    return eta
