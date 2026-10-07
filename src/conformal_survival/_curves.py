"""Survival curves on a shared time grid, and adapters for fitted survival models.

Every model the package uses (survival model for T, censoring model for C) is
reduced to a matrix of survival probabilities S(t | x) on a grid of times. The
conformal procedures only ever need three operations on those curves:
quantiles, CDF values and survival values at given times.
"""
from __future__ import annotations

import numpy as np


class SurvivalCurves:
    """Right-continuous step survival functions for n individuals.

    ``surv[i, j]`` is S(t | x_i) for t in [times[j], times[j + 1]); S = 1 before
    ``times[0]``.
    """

    def __init__(self, times, surv):
        times = np.asarray(times, dtype=float)
        surv = np.atleast_2d(np.asarray(surv, dtype=float))
        if surv.shape[1] != times.shape[0]:
            raise ValueError("surv must have one column per time point")
        if np.any(np.diff(times) < 0):
            raise ValueError("times must be sorted")
        self.times = times
        self.surv = surv

    def __len__(self):
        return self.surv.shape[0]

    def survival(self, t):
        """S(t_i | x_i) for each row i. ``t`` is a scalar or an array of length n."""
        t = np.broadcast_to(np.asarray(t, dtype=float), (len(self),))
        idx = np.searchsorted(self.times, t, side="right") - 1
        out = np.ones(len(self))
        ok = idx >= 0
        out[ok] = self.surv[np.nonzero(ok)[0], idx[ok]]
        return out

    def cdf(self, t):
        return 1.0 - self.survival(t)

    def quantile(self, level):
        """inf{t : F(t | x_i) >= level_i}, i.e. inf{t : S(t | x_i) <= 1 - level_i}.

        Returns 0 where level <= 0 and +inf where the curve never drops that far
        (the quantile lies beyond the last grid time).
        """
        level = np.broadcast_to(np.asarray(level, dtype=float), (len(self),))
        hit = self.surv <= (1.0 - level)[:, None] + 1e-12
        first = np.argmax(hit, axis=1)
        q = np.where(hit.any(axis=1), self.times[first], np.inf)
        return np.where(level <= 0, 0.0, q)


def surv_array(time, event):
    """Structured array in the scikit-survival ``y`` format, without importing it."""
    y = np.empty(len(time), dtype=[("event", "?"), ("time", "<f8")])
    y["event"] = np.asarray(event, dtype=bool)
    y["time"] = np.asarray(time, dtype=float)
    return y


def predict_curves(model, X) -> SurvivalCurves:
    """Survival curves from a fitted model.

    Accepts any estimator whose ``predict_survival_function(X)`` returns either a
    :class:`SurvivalCurves` or a sequence of step functions with ``.x`` / ``.y``
    attributes (the scikit-survival convention).
    """
    out = model.predict_survival_function(X)
    if isinstance(out, SurvivalCurves):
        return out
    fns = list(out)
    times = np.asarray(fns[0].x, dtype=float)
    if all(len(f.x) == len(times) and np.array_equal(f.x, times) for f in fns):
        surv = np.vstack([np.asarray(f.y, dtype=float) for f in fns])
    else:
        times = np.unique(np.concatenate([np.asarray(f.x, dtype=float) for f in fns]))
        surv = np.vstack([np.asarray(f(np.clip(times, f.x[0], f.x[-1])), dtype=float) for f in fns])
        for i, f in enumerate(fns):
            surv[i, times < f.x[0]] = 1.0
    return SurvivalCurves(times, surv)


class KaplanMeier:
    """Covariate-free Kaplan-Meier estimator with the scikit-survival interface.

    Used as the default censoring model: it is correct whenever censoring is
    independent of the covariates, and it has no compiled dependencies.
    """

    def fit(self, X, y):
        time = np.asarray(y["time"], dtype=float)
        event = np.asarray(y["event"], dtype=bool)
        times = np.unique(time)
        at_risk = len(time) - np.searchsorted(np.sort(time), times, side="left")
        deaths = np.bincount(np.searchsorted(times, time[event]), minlength=len(times))
        self.times_ = times
        self.surv_ = np.cumprod(1.0 - deaths / at_risk)
        return self

    def predict_survival_function(self, X):
        n = len(X)
        return SurvivalCurves(self.times_, np.tile(self.surv_, (n, 1)))
