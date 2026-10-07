"""Real-data example: GBSG2 breast-cancer recurrence-free survival (686 patients).

The data are right-censored, so only the DR method applies. True survival times of
censored patients are unknown, so coverage is reported two ways:

* certified lower bound: because T >= observed time, the share of test patients
  with observed time >= L can only understate P(T >= L). Assumption-free but loose
  when censoring is heavy.
* IPCW estimate: P(T >= L) = E[1{observed time >= L} / G(L | X)] with
  G(t) = P(C >= t), estimated by Kaplan-Meier on the training fold. Unbiased if
  censoring is independent of X; the standard way to evaluate on real data.

  python real_data.py --splits 50
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.datasets import load_gbsg2
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis

from conformal_survival import DRConformalSurvivalLPB, KaplanMeier, predict_curves, surv_array

ALPHA = 0.1
RESULTS = Path(__file__).resolve().parent / "results"


def load():
    X, y = load_gbsg2()
    X = pd.get_dummies(X, drop_first=True).astype(float)
    return X.to_numpy(), y["time"].astype(float), y["cens"].astype(bool), list(X.columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=50)
    args = ap.parse_args()
    X, time, event, cols = load()
    print(f"GBSG2: n={len(time)}, p={X.shape[1]}, events={event.mean():.0%}")

    rows = []
    for s in range(args.splits):
        rng = np.random.default_rng(s)
        perm = rng.permutation(len(time))
        test, train = perm[: len(time) // 4], perm[len(time) // 4:]
        G = KaplanMeier().fit(None, surv_array(time[train], ~event[train]))
        bounds = {}
        cox = CoxPHSurvivalAnalysis(alpha=1e-2).fit(X[train], surv_array(time[train], event[train]))
        bounds["Cox quantile (uncalibrated)"] = predict_curves(cox, X[test]).quantile(ALPHA)
        for name, model in [("DR conformal, Cox", CoxPHSurvivalAnalysis(alpha=1e-2)),
                            ("DR conformal, RSF", RandomSurvivalForest(
                                n_estimators=200, min_samples_leaf=15, random_state=s))]:
            m = DRConformalSurvivalLPB(model, censoring_model=CoxPHSurvivalAnalysis(alpha=1e-2),
                                       alpha=ALPHA, c0="auto", random_state=s)
            bounds[name] = m.fit(X[train], time[train], event[train]).predict(X[test])
        for name, L in bounds.items():
            L = np.minimum(L, 1e9)
            # G(L-) = P(C >= L): evaluate the KM curve just below L
            g = predict_curves(G, X[test]).survival(np.nextafter(L, 0))
            ipcw = np.mean(np.where(time[test] >= L, 1 / np.maximum(g, 0.05), 0.0))
            rows.append({"split": s, "method": name,
                         "certified_coverage_lower_bound": float(np.mean(time[test] >= L)),
                         "ipcw_coverage": float(ipcw),
                         "median_lpb_days": float(np.median(L))})

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "real_gbsg2.csv", index=False)
    print(df.groupby("method")[["certified_coverage_lower_bound", "ipcw_coverage", "median_lpb_days"]]
          .agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
