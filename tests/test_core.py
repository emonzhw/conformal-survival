import numpy as np
import pytest
from scipy import stats

from conformal_survival import (
    ConformalSurvivalLPB,
    DRConformalSurvivalLPB,
    KaplanMeier,
    SurvivalCurves,
    impute_censoring_times,
    surv_array,
    weighted_quantile_with_inf,
)
from conformal_survival.simulate import OracleExponentialCensoring, simulate


def brute_force_quantile(scores, weights, w_test, level):
    """sup{z : Q(Z <= z) < level}, evaluated directly from the definition."""
    total = weights.sum() + w_test
    candidates = np.sort(np.unique(scores))
    for z in candidates:
        if weights[scores <= z].sum() / total >= level:
            return z
    return np.inf


def test_weighted_quantile_matches_definition():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = rng.integers(1, 40)
        scores = rng.normal(size=n).round(1)          # rounding creates ties
        weights = rng.exponential(size=n)
        w_test = rng.exponential(size=5)
        level = rng.uniform(0.5, 0.99)
        got = weighted_quantile_with_inf(scores, weights, w_test, level)
        want = [brute_force_quantile(scores, weights, w, level) for w in w_test]
        np.testing.assert_allclose(got, want)


def test_unweighted_quantile_is_standard_split_conformal():
    rng = np.random.default_rng(1)
    scores = rng.normal(size=99)
    alpha = 0.1
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))   # usual conformal rank
    got = weighted_quantile_with_inf(scores, np.ones(99), np.ones(1), 1 - alpha)[0]
    assert got == np.sort(scores)[k - 1]


def test_quantile_is_infinite_when_test_weight_dominates():
    got = weighted_quantile_with_inf(np.arange(5.0), np.ones(5), np.array([1e9, np.inf]), 0.9)
    assert np.all(np.isinf(got))


def test_survival_curves_operations():
    curves = SurvivalCurves([1.0, 2.0, 3.0], [[0.9, 0.5, 0.1], [1.0, 1.0, 0.95]])
    np.testing.assert_allclose(curves.survival(0.5), [1.0, 1.0])
    np.testing.assert_allclose(curves.survival(2.5), [0.5, 1.0])
    np.testing.assert_allclose(curves.cdf(np.array([1.0, 3.0])), [0.1, 0.05])
    np.testing.assert_allclose(curves.quantile(0.5), [2.0, np.inf])
    np.testing.assert_allclose(curves.quantile(np.array([0.0, 0.05])), [0.0, 3.0])


def test_kaplan_meier_hand_example():
    # times 1, 2+, 3, 4 (+ = censored): S(1) = 3/4, S(3) = 3/4 * 1/2, S(4) = 0
    km = KaplanMeier().fit(None, surv_array([1, 2, 3, 4], [1, 0, 1, 1]))
    curves = km.predict_survival_function(np.zeros((1, 1)))
    np.testing.assert_allclose(curves.survival(np.array([1.0])), [0.75])
    np.testing.assert_allclose(curves.survival(np.array([3.5])), [0.375])
    np.testing.assert_allclose(curves.survival(np.array([4.0])), [0.0])


def test_imputation_recovers_memoryless_censoring():
    """With C ~ Exp(rate) known, C - t | C > t is again Exp(rate)."""
    rng = np.random.default_rng(2)
    n = 4000
    X = rng.uniform(0, 4, size=(n, 1))
    time = rng.uniform(0.1, 3.0, size=n)
    event = np.ones(n, dtype=bool)
    model = OracleExponentialCensoring("uvt_homo", "independent", t_max=80, n_grid=20000)
    C = impute_censoring_times(model, X, time, event, rng)
    excess = C - time
    assert np.all(excess > 0)
    p = stats.kstest(excess, stats.expon(scale=1 / 0.4).cdf).pvalue
    assert p > 1e-3


def test_censored_units_keep_their_observed_time():
    X = np.zeros((3, 1))
    C = impute_censoring_times(OracleExponentialCensoring(), X, np.array([1.0, 2.0, 3.0]),
                               np.array([False, True, False]), 0)
    assert C[0] == 1.0 and C[2] == 3.0 and C[1] > 2.0


class TinyQuantileModel:
    """A deliberately misspecified survival model: same lognormal curve for everyone.

    Fast, has no dependencies, and being wrong is the point: conformal calibration
    must still deliver coverage.
    """

    def get_params(self, deep=True):
        return {}

    def set_params(self, **p):
        return self

    def fit(self, X, y):
        t = y["time"][y["event"]]
        self.mu_, self.sd_ = np.log(t).mean(), np.log(t).std()
        return self

    def predict_survival_function(self, X):
        times = np.geomspace(1e-2, 1e3, 2000)
        s = 1 - stats.norm.cdf((np.log(times) - self.mu_) / self.sd_)
        return SurvivalCurves(times, np.tile(s, (len(X), 1)))


@pytest.mark.parametrize("score", ["cqr", "cdr"])
def test_type1_coverage_with_independent_censoring(score):
    """Exact finite-sample guarantee when weights are constant (Proposition 1)."""
    covs = []
    for rep in range(60):
        d = simulate("uvt_hetero", 600, "independent", random_state=rep)
        test = simulate("uvt_hetero", 2000, "independent", random_state=10_000 + rep)
        m = ConformalSurvivalLPB(TinyQuantileModel(), alpha=0.1, c0=3.0, score=score,
                                 random_state=rep).fit(d.X, d.time, d.event, d.C)
        covs.append(np.mean(test.T >= m.predict(test.X)))
    assert np.mean(covs) >= 0.9 - 0.01


def test_type1_oracle_weights_fix_dependent_censoring():
    covs = []
    oracle = OracleExponentialCensoring("uvt_hetero", "dependent")
    for rep in range(60):
        d = simulate("uvt_hetero", 600, "dependent", random_state=rep)
        test = simulate("uvt_hetero", 2000, "dependent", random_state=10_000 + rep)
        m = ConformalSurvivalLPB(TinyQuantileModel(), censoring_model=oracle, alpha=0.1, c0=3.0,
                                 random_state=rep).fit(d.X, d.time, d.event, d.C)
        covs.append(np.mean(test.T >= m.predict(test.X)))
    assert np.mean(covs) >= 0.9 - 0.01


def test_dr_with_oracle_censoring_covers():
    covs = []
    oracle = OracleExponentialCensoring("uvt_hetero", "dependent")
    for rep in range(40):
        d = simulate("uvt_hetero", 600, "dependent", random_state=rep)
        test = simulate("uvt_hetero", 2000, "dependent", random_state=10_000 + rep)
        m = DRConformalSurvivalLPB(TinyQuantileModel(), censoring_model=oracle, alpha=0.1, c0=3.0,
                                   random_state=rep).fit(d.X, d.time, d.event)
        covs.append(np.mean(test.T >= m.predict(test.X)))
    assert np.mean(covs) >= 0.9 - 0.015


def test_bounds_are_capped_and_nonnegative():
    d = simulate("uvt_homo", 500, "independent", random_state=0)
    m = ConformalSurvivalLPB(TinyQuantileModel(), c0=2.0, random_state=0).fit(d.X, d.time, d.event, d.C)
    L = m.predict(d.X)
    assert np.all(L >= 0) and np.all(L <= 2.0)


def test_auto_c0_is_chosen_from_grid():
    d = simulate("uvt_homo", 800, "independent", random_state=0)
    m = ConformalSurvivalLPB(TinyQuantileModel(), c0="auto", random_state=0).fit(d.X, d.time, d.event, d.C)
    assert m.c0_ in m.c0_grid_
    assert len(m.c0_scores_) == len(m.c0_grid_)


def test_ignoring_dependent_censoring_breaks_coverage():
    """Negative control: in the strong-shift setting, unweighted calibration
    (Kaplan-Meier censoring model) under-covers while oracle weights do not.
    This checks that the coverage tests above have power to detect a wrong method."""
    def mean_cov(cens_model):
        covs = []
        for rep in range(40):
            d = simulate("shift", 600, "dependent", random_state=rep)
            test = simulate("shift", 2000, "dependent", random_state=10_000 + rep)
            m = ConformalSurvivalLPB(TinyQuantileModel(), censoring_model=cens_model, alpha=0.1,
                                     c0=3.0, random_state=rep).fit(d.X, d.time, d.event, d.C)
            covs.append(np.mean(test.T >= m.predict(test.X)))
        return np.mean(covs)

    assert mean_cov(OracleExponentialCensoring("shift", "dependent")) >= 0.89
    assert mean_cov(None) < 0.85
