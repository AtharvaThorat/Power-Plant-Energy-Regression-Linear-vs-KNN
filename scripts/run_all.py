"""Run every experiment and write results/results.json.

    python scripts/run_all.py            # full run, about a minute
    python scripts/run_all.py --quick    # skips the 5x2 cross-validation
"""
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ccpp import (BASELINE_FORMULA, PREDICTORS, backward_eliminate, fit_ols, five_by_two_cv,  # noqa: E402
                  knn_sweep, load, mse, split)

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def main(quick=False):
    os.makedirs(OUT, exist_ok=True)
    df = load()
    res = {"n_rows": len(df), "summary": df.describe().round(3).to_dict(),
           "correlation": df.corr().round(4).to_dict()}

    # one predictor at a time, with studentized-residual outliers (|t| > 3)
    uni = {}
    for c in PREDICTORS:
        m = fit_ols(f"PE ~ {c}", df)
        t = m.get_influence().resid_studentized_internal
        clean = fit_ols(f"PE ~ {c}", df[np.abs(t) <= 3])
        uni[c] = {"slope": m.params[c], "intercept": m.params["Intercept"], "r2": m.rsquared,
                  "p": m.pvalues[c], "n_outliers": int((np.abs(t) > 3).sum()),
                  "slope_no_outliers": clean.params[c], "r2_no_outliers": clean.rsquared}
    res["univariate"] = uni

    # all four together
    mlr = fit_ols(BASELINE_FORMULA, df)
    res["multiple"] = {"coef": mlr.params.to_dict(), "p": mlr.pvalues.to_dict(), "r2": mlr.rsquared,
                       "adj_r2": mlr.rsquared_adj, "condition_number": float(mlr.condition_number)}
    res["rh_given_at"] = fit_ols("PE ~ RH + AT", df).params["RH"]

    # cubic terms, one predictor at a time
    res["cubic"] = {}
    for c in PREDICTORS:
        m = fit_ols(f"PE ~ {c} + I({c}**2) + I({c}**3)", df)
        res["cubic"][c] = {"r2": m.rsquared, "p_sq": m.pvalues[f"I({c} ** 2)"], "p_cube": m.pvalues[f"I({c} ** 3)"]}

    # all pairwise interactions
    inter = fit_ols("PE ~ (AT + V + AP + RH)**2", df)
    res["interactions"] = {"r2": inter.rsquared,
                           "p": {k: v for k, v in inter.pvalues.items() if ":" in k},
                           "coef": {k: v for k, v in inter.params.items() if ":" in k}}

    # 70/30 split: baseline vs. backward-eliminated model vs. KNN
    tr, te = split(df)
    base = fit_ols(BASELINE_FORMULA, tr)
    formula, removed = backward_eliminate(tr)
    enh = fit_ols(formula, tr)
    res["holdout"] = {
        "n_train": len(tr), "n_test": len(te),
        "baseline": {"train_mse": mse(tr.PE, base.predict(tr)), "test_mse": mse(te.PE, base.predict(te))},
        "enhanced": {"formula": formula, "removed": removed, "n_terms": len(enh.params) - 1,
                     "train_mse": mse(tr.PE, enh.predict(tr)), "test_mse": mse(te.PE, enh.predict(te))},
    }
    for scale in (True, False):
        s = knn_sweep(tr, te, scale=scale)
        best = s.loc[s["test_mse"].idxmin()]
        res["holdout"]["knn_scaled" if scale else "knn_raw"] = {
            "best_k": int(best.k), "test_mse": float(best.test_mse), "train_mse_at_best": float(best.train_mse),
            "curve": s.to_dict(orient="list")}

    # 5 x 2 CV: nothing is tuned on the data it is scored on
    if not quick:
        cv = five_by_two_cv(df, formula, extra_models={
            "gradient_boosting": HistGradientBoostingRegressor(random_state=0)})
        models = ["linear", "enhanced_linear", "knn_raw", "knn_scaled", "gradient_boosting"]
        res["cv5x2"] = {
            "folds": cv.to_dict(orient="list"),
            "mean": {m: float(cv[m].mean()) for m in models},
            "std": {m: float(cv[m].std()) for m in models},
            "knn_beats_linear_folds": int((cv["knn_scaled"] < cv["enhanced_linear"]).sum()),
            "knn_k_range": [int(cv["knn_scaled_k"].min()), int(cv["knn_scaled_k"].max())],
        }

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)

    h = res["holdout"]
    print(f"baseline   test MSE {h['baseline']['test_mse']:.2f}")
    print(f"enhanced   test MSE {h['enhanced']['test_mse']:.2f}  (removed {[t for t, _ in removed]})")
    print(f"KNN scaled test MSE {h['knn_scaled']['test_mse']:.2f} at k = {h['knn_scaled']['best_k']}")
    if not quick:
        for m, v in res["cv5x2"]["mean"].items():
            print(f"5x2 CV {m:<18} {v:6.2f} ± {res['cv5x2']['std'][m]:.2f}")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
