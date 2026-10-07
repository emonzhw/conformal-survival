"""Conformal lower prediction bounds (LPBs) for survival times.

Two estimators:

* :class:`ConformalSurvivalLPB` -- Candes, Lei & Ren (2023), "Conformalized
  survival analysis", JRSS-B. Type-I censoring: the censoring time C is observed
  for every unit (e.g. administrative end of follow-up).
* :class:`DRConformalSurvivalLPB` -- Sesia & Svetnik (2025), "Doubly robust
  conformalized survival analysis with right-censored data", ICML. Ordinary right
  censoring: C is only seen for censored units, so it is imputed for the rest
  from a censoring model before applying the Candes et al. procedure.

Both return L(x) with P(T >= L(X)) >= 1 - alpha. The guarantee is exact in finite
samples when the censoring weights are known; it holds approximately when either
the censoring model or the survival model is estimated well.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, clone

from ._curves import KaplanMeier, predict_curves, surv_array
from ._weighted import weighted_quantile_with_inf

_MIN_PROB = 1e-12


def _as_rng(random_state):
    return random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)


class _BaseConformalLPB(BaseEstimator):
    def __init__(self, survival_model, censoring_model=None, alpha=0.1, c0="auto",
                 score="cqr", cal_size=0.5, c0_grid=None, holdout_size=0.25, random_state=None):
        self.survival_model = survival_model
        self.censoring_model = censoring_model
        self.alpha = alpha
        self.c0 = c0
        self.score = score
        self.cal_size = cal_size
        self.c0_grid = c0_grid
        self.holdout_size = holdout_size
        self.random_state = random_state

    # --- hooks implemented by subclasses -------------------------------------
    def _fit_models(self, X, time, event, censor_time):
        raise NotImplementedError

    def _calibration_censor_times(self, X, time, event, censor_time, rng):
        raise NotImplementedError

    def _grid_source(self, time, censor_time):
        raise NotImplementedError

    # --- shared machinery ------------------------------------------------------
    def _check(self, X, time, event, censor_time):
        X = np.asarray(X, dtype=float)
        time = np.asarray(time, dtype=float)
        event = np.asarray(event, dtype=bool)
        n = X.shape[0]
        if time.shape != (n,) or event.shape != (n,):
            raise ValueError("time and event must be 1-D with one entry per row of X")
        if censor_time is not None:
            censor_time = np.asarray(censor_time, dtype=float)
            if censor_time.shape != (n,):
                raise ValueError("censor_time must have one entry per row of X")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if self.score not in ("cqr", "cdr"):
            raise ValueError("score must be 'cqr' or 'cdr'")
        return X, time, event, censor_time

    def _censor_prob(self, models, X, c0):
        """c_hat(x) = P(C >= c0 | X = x) from the censoring model."""
        return predict_curves(models["censoring"], X).survival(np.full(len(X), c0))

    def _scores(self, models, X, y, c0):
        curves = predict_curves(models["survival"], X)
        if self.score == "cqr":
            # V = q_alpha(x; c0) - y, with q_alpha(x; c0) = q_alpha(x) ^ c0
            return np.minimum(curves.quantile(self.alpha), c0) - y
        # V = alpha - F(y | x) for the distribution of T ^ c0, which is 1 at y = c0
        F = np.where(y >= c0, 1.0, curves.cdf(y))
        return self.alpha - F

    def _calibrate(self, models, X, time, C, c0):
        keep = C >= c0
        y = np.minimum(time[keep], c0)
        scores = self._scores(models, X[keep], y, c0)
        weights = 1.0 / np.maximum(self._censor_prob(models, X[keep], c0), _MIN_PROB)
        return {"c0": c0, "scores": scores, "weights": weights, "n_used": int(keep.sum())}

    def _bound(self, models, cal, X):
        c0 = cal["c0"]
        w = 1.0 / np.maximum(self._censor_prob(models, X, c0), _MIN_PROB)
        eta = weighted_quantile_with_inf(cal["scores"], cal["weights"], w, 1 - self.alpha)
        curves = predict_curves(models["survival"], X)
        if self.score == "cqr":
            with np.errstate(invalid="ignore"):
                lpb = np.minimum(curves.quantile(self.alpha), c0) - eta
        else:
            lpb = curves.quantile(self.alpha - eta)
        lpb = np.minimum(lpb, c0)
        return np.where(np.isfinite(lpb), np.maximum(lpb, 0.0), 0.0)

    def _select_c0(self, X, time, event, censor_time, rng):
        """Candes et al. Section 3.3: pick c0 on the training fold only.

        Split the training fold into fit / calibration / holdout parts, run the
        procedure for every candidate c0 and keep the one with the largest mean
        LPB on the holdout part. The calibration fold is never touched.
        """
        if self.c0_grid is not None:
            grid = np.asarray(self.c0_grid, dtype=float)
        else:
            grid = np.unique(np.quantile(self._grid_source(time, censor_time), np.linspace(0.1, 0.9, 9)))
        n = len(time)
        perm = rng.permutation(n)
        n_hold = int(round(self.holdout_size * n))
        hold, rest = perm[:n_hold], perm[n_hold:]
        n_cal = int(round(self.cal_size * len(rest)))
        cal, fit = rest[:n_cal], rest[n_cal:]
        ct = None if censor_time is None else censor_time
        models = self._fit_models(X[fit], time[fit], event[fit], None if ct is None else ct[fit])
        C_cal = self._calibration_censor_times(
            models, X[cal], time[cal], event[cal], None if ct is None else ct[cal], rng)
        means = []
        for c0 in grid:
            calib = self._calibrate(models, X[cal], time[cal], C_cal, c0)
            means.append(self._bound(models, calib, X[hold]).mean())
        self.c0_grid_ = grid
        self.c0_scores_ = np.array(means)
        return float(grid[int(np.argmax(means))])

    def _fit(self, X, time, event, censor_time):
        X, time, event, censor_time = self._check(X, time, event, censor_time)
        rng = _as_rng(self.random_state)
        perm = rng.permutation(len(time))
        n_cal = int(round(self.cal_size * len(time)))
        cal, tr = perm[:n_cal], perm[n_cal:]

        def part(a, idx):
            return None if a is None else a[idx]

        if isinstance(self.c0, str):
            if self.c0 != "auto":
                raise ValueError("c0 must be a number or 'auto'")
            c0 = self._select_c0(X[tr], time[tr], event[tr], part(censor_time, tr), rng)
        else:
            c0 = float(self.c0)
        self.models_ = self._fit_models(X[tr], time[tr], event[tr], part(censor_time, tr))
        C_cal = self._calibration_censor_times(
            self.models_, X[cal], time[cal], event[cal], part(censor_time, cal), rng)
        self.calibration_ = self._calibrate(self.models_, X[cal], time[cal], C_cal, c0)
        self.c0_ = c0
        self.n_calibration_used_ = self.calibration_["n_used"]
        return self

    def predict(self, X):
        """Lower prediction bounds L(x) for the survival times of new units."""
        X = np.asarray(X, dtype=float)
        return self._bound(self.models_, self.calibration_, X)


class ConformalSurvivalLPB(_BaseConformalLPB):
    """Conformalized survival analysis under type-I censoring (Candes, Lei & Ren, 2023).

    Parameters
    ----------
    survival_model : estimator with the scikit-survival API
        ``fit(X, y)`` with structured ``y`` and ``predict_survival_function(X)``.
        Fitted to (time, event) on the training fold.
    censoring_model : estimator with the same API, optional
        Model for C given X, fitted to the fully observed censoring times. Default:
        Kaplan-Meier, i.e. censoring independent of X (weights are then constant and
        the guarantee is exact).
    alpha : float
        Miscoverage level; the LPB targets P(T >= L(X)) >= 1 - alpha.
    c0 : float or "auto"
        Censoring threshold. "auto" selects it on the training fold.
    score : {"cqr", "cdr"}
        Conformalized quantile regression or distribution regression scores.
    cal_size : float
        Fraction of the data used for calibration.
    """

    def fit(self, X, time, event, censor_time):
        """Fit on (X, observed time, event indicator, censoring time C for every unit)."""
        if censor_time is None:
            raise ValueError("type-I censoring needs censor_time; use DRConformalSurvivalLPB otherwise")
        return self._fit(X, time, event, censor_time)

    def _fit_models(self, X, time, event, censor_time):
        surv = clone(self.survival_model).fit(X, surv_array(time, event))
        cens_model = KaplanMeier() if self.censoring_model is None else clone(self.censoring_model)
        cens = cens_model.fit(X, surv_array(censor_time, np.ones_like(censor_time, dtype=bool)))
        return {"survival": surv, "censoring": cens}

    def _calibration_censor_times(self, models, X, time, event, censor_time, rng):
        return censor_time

    def _grid_source(self, time, censor_time):
        return censor_time


class DRConformalSurvivalLPB(_BaseConformalLPB):
    """Doubly robust conformalized survival analysis under right censoring
    (Sesia & Svetnik, 2025), fixed-cutoff version.

    Censoring times of calibration units that had an event are imputed by sampling
    from the censoring model conditional on C > observed time, which turns the data
    into a synthetic type-I-censored sample. The Candes et al. procedure is then
    applied, and the result is capped at the survival model's own alpha-quantile,
    which is what gives the double robustness property.

    Parameters are as in :class:`ConformalSurvivalLPB`; the censoring model is fitted
    to (time, 1 - event) on the training fold.
    """

    def fit(self, X, time, event):
        """Fit on right-censored data: (X, observed time, event indicator)."""
        return self._fit(X, time, event, None)

    def _fit_models(self, X, time, event, censor_time):
        surv = clone(self.survival_model).fit(X, surv_array(time, event))
        cens_model = KaplanMeier() if self.censoring_model is None else clone(self.censoring_model)
        cens = cens_model.fit(X, surv_array(time, ~event))
        return {"survival": surv, "censoring": cens}

    def _calibration_censor_times(self, models, X, time, event, censor_time, rng):
        return impute_censoring_times(models["censoring"], X, time, event, rng)

    def _grid_source(self, time, censor_time):
        return time

    def _bound(self, models, cal, X):
        lpb = super()._bound(models, cal, X)
        q = predict_curves(models["survival"], X).quantile(self.alpha)
        return np.minimum(lpb, q)


def impute_censoring_times(censoring_model, X, time, event, random_state=None):
    """Algorithm 1 of Sesia & Svetnik (2025).

    Censored units (event = 0) keep C = observed time. For units with an event,
    C is drawn from the fitted distribution of C | X, conditional on C > observed
    time, by inverse-transform sampling: S(C) = U * S(observed time), U ~ Unif(0, 1).
    Draws beyond the last time on the model's grid are returned as +inf.
    """
    rng = _as_rng(random_state)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    C = time.copy()
    idx = np.nonzero(event)[0]
    if idx.size == 0:
        return C
    curves = predict_curves(censoring_model, np.asarray(X, dtype=float)[idx])
    s_obs = curves.survival(time[idx])
    target = rng.uniform(size=idx.size) * s_obs
    grid = curves.times
    after = grid[None, :] > time[idx, None]
    hit = after & (curves.surv <= target[:, None])
    first = np.argmax(hit, axis=1)
    draw = np.where(hit.any(axis=1), grid[first], np.inf)
    # If the model gives no mass beyond the observed time, fall back to that time.
    C[idx] = np.where(s_obs > 0, draw, time[idx])
    return C
