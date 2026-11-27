"""Model fitting and evaluation."""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from ccpp.data import PREDICTORS, TARGET

BASELINE_FORMULA = "PE ~ AT + V + AP + RH"
FULL_FORMULA = "PE ~ (AT + V + AP + RH)**2 + I(AT ** 2) + I(V ** 2) + I(AP ** 2) + I(RH ** 2)"


def mse(y, y_hat):
    return float(np.mean((np.asarray(y) - np.asarray(y_hat)) ** 2))


def fit_ols(formula, data):
    return smf.ols(formula, data=data).fit()


def _mentions(term, main):
    """True if an interaction ('AT:V') or polynomial ('I(AT ** 2)') term involves a main effect."""
    if ":" in term:
        return main in term.split(":")
    return term.startswith("I(") and term[2:].split(" ")[0] == main


def backward_eliminate(data, formula=FULL_FORMULA, alpha=0.05):
    """Drop the least significant term until every term has p <= alpha.

    Hierarchy is respected: a main effect stays while any interaction or power of it remains.
    Returns (final_formula, [(removed_term, p_value), ...]).
    """
    terms = [t for t in fit_ols(formula, data).params.index if t != "Intercept"]
    removed = []
    while True:
        p = fit_ols(f"{TARGET} ~ " + " + ".join(terms), data).pvalues.drop("Intercept")
        candidates = [
            (t, pv) for t, pv in p.sort_values(ascending=False).items()
            if pv > alpha and not (t in PREDICTORS and any(_mentions(o, t) for o in terms if o != t))
        ]
        if not candidates:
            return f"{TARGET} ~ " + " + ".join(terms), removed
        t, pv = candidates[0]
        terms.remove(t)
        removed.append((t, float(pv)))


def knn_sweep(train, test, ks=range(1, 101), scale=True):
    """Train and test MSE of KNN regression for every k, with optional min-max scaling."""
    Xtr, Xte = train[PREDICTORS].to_numpy(), test[PREDICTORS].to_numpy()
    if scale:
        sc = MinMaxScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    rows = []
    for k in ks:
        m = KNeighborsRegressor(n_neighbors=int(k)).fit(Xtr, train[TARGET])
        rows.append((int(k), mse(train[TARGET], m.predict(Xtr)), mse(test[TARGET], m.predict(Xte))))
    return pd.DataFrame(rows, columns=["k", "train_mse", "test_mse"])


def five_by_two_cv(df, enhanced_formula, k_grid=range(1, 31), seeds=range(5), extra_models=None):
    """5 x 2-fold cross-validation, the protocol the dataset was published for.

    KNN's k is tuned by an inner 5-fold CV on each training half, so the test half is never
    used for selection. Returns one row per fold with test MSE for every model.
    """
    extra_models = extra_models or {}
    rows = []
    for seed in seeds:
        for fold, (a, b) in enumerate(KFold(2, shuffle=True, random_state=seed).split(df)):
            A, B = df.iloc[a], df.iloc[b]
            r = {"seed": seed, "fold": fold}
            r["linear"] = mse(B[TARGET], fit_ols(BASELINE_FORMULA, A).predict(B))
            r["enhanced_linear"] = mse(B[TARGET], fit_ols(enhanced_formula, A).predict(B))
            for name, est, key in [
                ("knn_scaled", make_pipeline(MinMaxScaler(), KNeighborsRegressor()), "kneighborsregressor__n_neighbors"),
                ("knn_raw", KNeighborsRegressor(), "n_neighbors"),
            ]:
                g = GridSearchCV(est, {key: list(k_grid)}, cv=5, scoring="neg_mean_squared_error").fit(A[PREDICTORS], A[TARGET])
                r[name] = mse(B[TARGET], g.predict(B[PREDICTORS]))
                r[name + "_k"] = g.best_params_[key]
            for name, est in extra_models.items():
                r[name] = mse(B[TARGET], est.fit(A[PREDICTORS], A[TARGET]).predict(B[PREDICTORS]))
            rows.append(r)
    return pd.DataFrame(rows)
