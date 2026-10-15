"""Coverage and efficiency of survival LPBs over repeated simulations.

Reproduces the design of Candes, Lei & Ren (2023), Section 4 (Figure 1): train on
n units, evaluate on n fresh test units, target 90% coverage of the *uncensored*
survival time T, repeat. Adds the right-censored DR method (Sesia & Svetnik, 2025)
and a dependent-censoring stress test.

  python coverage.py --reps 100 --settings uvt_homo uvt_hetero shift
  python coverage.py --reps 2 --n 600 --quick      # smoke test

Writes results/coverage_<setting>.csv with one row per (rep, method).
"""
import argparse
import time as clock
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis

from conformal_survival import (ConformalSurvivalLPB, DRConformalSurvivalLPB, predict_curves,
                                surv_array, weighted_quantile_with_inf)
from conformal_survival.simulate import simulate

ALPHA = 0.1
RESULTS = Path(__file__).resolve().parent / "results"
# The paper's settings use independent censoring; "shift" is our dependent-censoring stress test.
CENSORING = {"uvt_homo": "independent", "uvt_hetero": "independent",
             "mvt_homo": "independent", "mvt_hetero": "independent", "shift": "dependent"}


def cox():
    return CoxPHSurvivalAnalysis(alpha=1e-4)   # tiny ridge penalty for numerical stability


N_TREES = 200


def rsf(seed, quick):
    return RandomSurvivalForest(n_estimators=50 if quick else N_TREES, min_samples_leaf=20,
                                max_features="sqrt", n_jobs=1, random_state=seed)


def naive_cqr(model, X, time, event, X_test, rng):
    """Split CQR applied to the censored time T~ as if it were T (overly conservative)."""
    perm = rng.permutation(len(time))
    cal, tr = perm[: len(time) // 2], perm[len(time) // 2:]
    fitted = model.fit(X[tr], surv_array(time[tr], event[tr]))
    q_cal = np.minimum(predict_curves(fitted, X[cal]).quantile(ALPHA), 1e12)
    eta = weighted_quantile_with_inf(q_cal - time[cal], np.ones(len(cal)), np.ones(1), 1 - ALPHA)[0]
    q = np.minimum(predict_curves(fitted, X_test).quantile(ALPHA), 1e12)
    return np.maximum(q - eta, 0.0)


def one_rep(setting, rep, n, quick):
    cens = CENSORING[setting]
    d = simulate(setting, n, cens, random_state=rep)
    test = simulate(setting, n, cens, random_state=1_000_000 + rep)
    oracle_q = test.true_quantile(ALPHA)
    rng = np.random.default_rng(rep)
    bounds = {}

    # Uncalibrated model quantiles, fitted on all training data.
    for name, model in [("Cox quantile", cox()), ("RSF quantile", rsf(rep, quick))]:
        fitted = model.fit(d.X, surv_array(d.time, d.event))
        bounds[name] = np.minimum(predict_curves(fitted, test.X).quantile(ALPHA), 1e12)

    bounds["Naive CQR"] = naive_cqr(rsf(rep, quick), d.X, d.time, d.event, test.X, rng)

    # Type-I methods (need C for every unit); censoring model: Cox on C.
    for name, model, score in [("Candes CQR-Cox", cox(), "cqr"),
                               ("Candes CQR-RSF", rsf(rep, quick), "cqr"),
                               ("Candes CDR-RSF", rsf(rep, quick), "cdr")]:
        m = ConformalSurvivalLPB(model, censoring_model=cox(), alpha=ALPHA, c0="auto",
                                 score=score, random_state=rep).fit(d.X, d.time, d.event, d.C)
        bounds[name] = m.predict(test.X)

    # Right-censored method (no access to C for units with an event).
    m = DRConformalSurvivalLPB(rsf(rep, quick), censoring_model=cox(), alpha=ALPHA, c0="auto",
                               random_state=rep).fit(d.X, d.time, d.event)
    bounds["DR-COSARC CQR-RSF"] = m.predict(test.X)

    rows = []
    for name, L in bounds.items():
        rows.append({"setting": setting, "rep": rep, "method": name,
                     "coverage": float(np.mean(test.T >= L)),
                     "lpb_over_oracle": float(np.median(L / oracle_q)),
                     "mean_lpb": float(np.mean(L))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=["uvt_homo", "uvt_hetero", "shift"])
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--quick", action="store_true", help="fewer trees, for smoke tests")
    ap.add_argument("--trees", type=int, default=200, help="trees per random survival forest")
    args = ap.parse_args()
    global N_TREES
    N_TREES = args.trees
    RESULTS.mkdir(exist_ok=True)

    for setting in args.settings:
        start = clock.time()
        out = Parallel(n_jobs=args.jobs, verbose=10)(
            delayed(one_rep)(setting, r, args.n, args.quick) for r in range(args.reps))
        df = pd.DataFrame([row for rows in out for row in rows])
        path = RESULTS / f"coverage_{setting}.csv"
        df.to_csv(path, index=False)
        summary = df.groupby("method")[["coverage", "lpb_over_oracle"]].agg(["mean", "std"])
        print(f"\n== {setting} ({CENSORING[setting]} censoring), {args.reps} reps, {args.trees} trees, "
              f"{clock.time() - start:.0f}s -> {path.name}")
        print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
