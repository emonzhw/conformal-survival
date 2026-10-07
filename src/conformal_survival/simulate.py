"""Simulated survival data with known ground truth, for checking coverage.

The survival settings follow Table 1 of Candes, Lei & Ren (2023): log T | X is
Gaussian with mean mu(X) and sd sigma(X). Censoring is either independent of X
(C ~ Exp(0.4), as in the paper) or depends on X through the first covariate,
which is the case where the conformal weights actually matter.

The extra "shift" setting (not in the paper) makes survival fall steeply with x
while long follow-up is concentrated at small x, so units with C >= c0 look very
different from the population. It is a stress test for the weighting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from ._curves import SurvivalCurves

SETTINGS = ("uvt_homo", "uvt_hetero", "mvt_homo", "mvt_hetero", "shift")


@dataclass
class SurvivalData:
    X: np.ndarray
    T: np.ndarray        # true survival time (unobserved in practice)
    C: np.ndarray        # censoring time
    time: np.ndarray     # observed time min(T, C)
    event: np.ndarray    # 1(T <= C)
    mu: np.ndarray
    sigma: np.ndarray

    def true_quantile(self, alpha):
        """Oracle LPB: the alpha-quantile of T | X."""
        return np.exp(self.mu + self.sigma * norm.ppf(alpha))


def _covariates(setting, n, rng):
    if setting == "shift":
        return rng.uniform(0, 1, size=(n, 1))
    if setting.startswith("uvt"):
        return rng.uniform(0, 4, size=(n, 1))
    return rng.uniform(-1, 1, size=(n, 100))


def _mu_sigma(setting, X):
    if setting == "shift":
        return 2.5 - 2.0 * X[:, 0], np.full(len(X), 0.6)
    if setting == "uvt_homo":
        return 2 + 0.37 * np.sqrt(X[:, 0]), np.full(len(X), 1.5)
    if setting == "uvt_hetero":
        return 2 + 0.37 * np.sqrt(X[:, 0]), 1 + X[:, 0] / 5
    mu = np.log(2) + 1 + 0.55 * (X[:, 0] ** 2 - X[:, 2] * X[:, 4])
    if setting == "mvt_homo":
        return mu, np.ones(len(X))
    if setting == "mvt_hetero":
        return mu, np.abs(X[:, 9]) + 1
    raise ValueError(f"unknown setting {setting!r}; choose from {SETTINGS}")


def censoring_rate(setting, X, censoring):
    """Exponential censoring rate for each unit."""
    if censoring == "independent":
        return np.full(len(X), 0.4)
    if censoring == "dependent":
        if setting == "shift":
            return 0.02 + 1.2 * X[:, 0]
        lo, hi = (0, 4) if setting.startswith("uvt") else (-1, 1)
        u = (X[:, 0] - lo) / (hi - lo)
        return 0.05 + 0.75 * u   # follow-up is long for small x1, short for large x1
    raise ValueError("censoring must be 'independent' or 'dependent'")


def simulate(setting="uvt_homo", n=1000, censoring="independent", random_state=None):
    rng = np.random.default_rng(random_state)
    X = _covariates(setting, n, rng)
    mu, sigma = _mu_sigma(setting, X)
    T = np.exp(mu + sigma * rng.standard_normal(n))
    C = rng.exponential(1 / censoring_rate(setting, X, censoring))
    return SurvivalData(X=X, T=T, C=C, time=np.minimum(T, C), event=(T <= C).astype(int),
                        mu=mu, sigma=sigma)


class OracleExponentialCensoring:
    """The true censoring distribution, usable wherever a censoring model is expected.

    ``fit`` is a no-op, so plugging this in gives the "known weights" version of
    the methods, the case where the coverage guarantee is exact.
    """

    def __init__(self, setting="uvt_homo", censoring="independent", t_max=200.0, n_grid=4000):
        self.setting = setting
        self.censoring = censoring
        self.t_max = t_max
        self.n_grid = n_grid

    def get_params(self, deep=True):
        return {"setting": self.setting, "censoring": self.censoring,
                "t_max": self.t_max, "n_grid": self.n_grid}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        return self

    def predict_survival_function(self, X):
        times = np.concatenate([[0.0], np.geomspace(1e-3, self.t_max, self.n_grid)])
        rate = censoring_rate(self.setting, np.asarray(X), self.censoring)
        return SurvivalCurves(times, np.exp(-rate[:, None] * times[None, :]))
