"""Pins the headline numbers. Runs in a few seconds (the 5x2 CV test uses one repeat)."""
import numpy as np
import pytest

from ccpp import BASELINE_FORMULA, backward_eliminate, fit_ols, five_by_two_cv, knn_sweep, load, mse, split


@pytest.fixture(scope="module")
def data():
    df = load()
    return df, *split(df)


def test_shape(data):
    df, tr, te = data
    assert df.shape == (9568, 5) and len(tr) == 6697 and len(te) == 2871


def test_multiple_regression(data):
    df, *_ = data
    m = fit_ols(BASELINE_FORMULA, df)
    assert m.rsquared == pytest.approx(0.9287, abs=1e-4)
    assert m.params["AT"] == pytest.approx(-1.9775, abs=1e-4)


def test_humidity_sign_flips_once_temperature_is_controlled(data):
    df, *_ = data
    assert fit_ols("PE ~ RH", df).params["RH"] > 0
    assert fit_ols(BASELINE_FORMULA, df).params["RH"] < 0
    assert df[["AT", "RH"]].corr().iloc[0, 1] < -0.5


def test_backward_elimination(data):
    _, tr, te = data
    formula, removed = backward_eliminate(tr)
    assert [t for t, _ in removed] == ["V:RH", "I(V ** 2)", "V:AP"]
    m = fit_ols(formula, tr)
    assert mse(te.PE, m.predict(te)) == pytest.approx(18.66, abs=0.01)


def test_knn_beats_linear_on_holdout(data):
    _, tr, te = data
    s = knn_sweep(tr, te, ks=range(1, 11))
    best = s.loc[s.test_mse.idxmin()]
    assert int(best.k) == 4 and best.test_mse == pytest.approx(14.29, abs=0.01)
    assert knn_sweep(tr, te, ks=[4], scale=False).test_mse[0] > best.test_mse   # scaling helps


def test_cv_ranking_holds(data):
    df, tr, _ = data
    formula, _ = backward_eliminate(tr)
    cv = five_by_two_cv(df, formula, k_grid=range(3, 10), seeds=[0])
    assert (cv["knn_scaled"] < cv["enhanced_linear"]).all()
    assert (cv["enhanced_linear"] < cv["linear"]).all()
