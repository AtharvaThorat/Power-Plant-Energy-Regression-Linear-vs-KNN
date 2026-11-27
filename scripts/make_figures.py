"""Render README figures from results/results.json (run scripts/run_all.py first)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ccpp import fit_ols, load, split  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
A = os.path.join(ROOT, "assets")
BG, TXT, MUTED, GRID = "#0d1117", "#e6edf3", "#8b949e", "#21262d"
BLUE, ORANGE, AQUA, YELLOW = "#338ced", "#e76624", "#00a888", "#d28e05"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TXT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlecolor": TXT, "axes.titlelocation": "left", "lines.linewidth": 2, "legend.frameon": False,
    "axes.axisbelow": True,
})


from matplotlib.colors import LinearSegmentedColormap
HEX = LinearSegmentedColormap.from_list("hex", ["#16263a", "#1f4f86", BLUE, "#9cc9f5"])


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(A, name), dpi=150)
    plt.close()


def main():
    os.makedirs(A, exist_ok=True)
    r = json.load(open(os.path.join(ROOT, "results", "results.json")))
    df = load()

    # 1. what drives output: each predictor vs PE, with linear and cubic fits
    labels = {"AT": "ambient temperature (°C)", "V": "exhaust vacuum (cm Hg)",
              "AP": "ambient pressure (mbar)", "RH": "relative humidity (%)"}
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharey=True)
    for ax, c in zip(axes, labels):
        ax.hexbin(df[c], df.PE, gridsize=38, cmap=HEX, mincnt=1, bins="log", linewidths=0)
        x = np.linspace(df[c].min(), df[c].max(), 200)
        u = r["univariate"][c]
        ax.plot(x, u["intercept"] + u["slope"] * x, color=ORANGE, lw=1.8, label="linear")
        cub = fit_ols(f"PE ~ {c} + I({c}**2) + I({c}**3)", df)
        import pandas as pd
        ax.plot(x, cub.predict(pd.DataFrame({c: x})), color=AQUA, lw=1.8, ls="--", label="cubic")
        ax.set_xlabel(labels[c])
        ax.set_title(f"{c}   R² = {u['r2']:.2f}", fontsize=12)
        ax.grid(False)
    axes[0].set_ylabel("net output PE (MW)")
    axes[0].legend(labelcolor=TXT, loc="lower left", fontsize=9.5)
    save("drivers.png")

    # 2. coefficient flip: one predictor alone vs. all four together
    fig, ax = plt.subplots(figsize=(10, 3.8))
    preds = ["AT", "V", "AP", "RH"]
    y = np.arange(len(preds))[::-1]
    for yi, c in zip(y, preds):
        a, b = r["univariate"][c]["slope"], r["multiple"]["coef"][c]
        col = ORANGE if np.sign(a) != np.sign(b) else MUTED
        ax.annotate("", xy=(b, yi), xytext=(a, yi), arrowprops=dict(arrowstyle="->", color=col, lw=2))
        ax.scatter([a], [yi], s=60, color="#2a3442", edgecolor=MUTED, zorder=3)
        ax.scatter([b], [yi], s=60, color=BLUE, zorder=3)
        ax.text(2.2, yi, f"{a:+.2f} → {b:+.2f}", va="center", color=col if col == ORANGE else TXT, fontsize=10)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, preds)
    ax.set_xlim(-2.6, 3.1)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("coefficient (MW per unit)   ·   grey = alone,  blue = with the other three")
    ax.set_title("Holding the others fixed shrinks most effects, and flips humidity")
    save("coefficient_flip.png")

    # 3. model comparison: 5x2 CV
    cv = r["cv5x2"]
    models = [("linear", "Linear, main effects"), ("enhanced_linear", "Linear + interactions + squares"),
              ("knn_raw", "KNN, raw features"), ("knn_scaled", "KNN, min-max scaled"),
              ("gradient_boosting", "Gradient boosting (reference)")]
    fig, ax = plt.subplots(figsize=(10, 4))
    y = np.arange(len(models))[::-1]
    best_main = min(cv["mean"][m] for m, _ in models[:4])
    for yi, (m, name) in zip(y, models):
        mean, sd = cv["mean"][m], cv["std"][m]
        col = BLUE if mean == best_main else ("#3a2a1e" if m == "gradient_boosting" else "#2a3442")
        ax.barh(yi, mean, height=0.55, color=col, edgecolor=ORANGE if m == "gradient_boosting" else col, ls="--" if m == "gradient_boosting" else "-")
        ax.errorbar(mean, yi, xerr=sd, color=TXT, capsize=3, lw=1)
        ax.text(mean + 0.6, yi, f"MSE {mean:.1f}  ·  RMSE {np.sqrt(mean):.2f} MW", va="center", color=TXT, fontsize=10)
    ax.set_yticks(y, [n for _, n in models])
    ax.set_xlim(0, 30)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("test MSE, mean ± sd over 5 × 2-fold cross-validation  ·  lower is better")
    ax.set_title("Model comparison, nothing tuned on its own test fold")
    save("model_comparison.png")

    # 4. KNN error vs k
    h = r["holdout"]
    fig, ax = plt.subplots(figsize=(10, 4))
    for key, col, lab in [("knn_scaled", BLUE, "scaled"), ("knn_raw", MUTED, "raw")]:
        cu = h[key]["curve"]
        ax.plot(cu["k"], cu["test_mse"], color=col, label=f"test, {lab} features")
        if key == "knn_scaled":
            ax.plot(cu["k"], cu["train_mse"], color=col, ls=":", label="train, scaled features")
    ax.axhline(h["enhanced"]["test_mse"], color=ORANGE, lw=1.5, ls="--")
    ax.text(12, h["enhanced"]["test_mse"] + 0.6, f"best linear model ({h['enhanced']['test_mse']:.2f})", color=ORANGE, ha="left", fontsize=10)
    b = h["knn_scaled"]
    ax.scatter([b["best_k"]], [b["test_mse"]], color=BLUE, s=50, zorder=3)
    ax.annotate(f"k = {b['best_k']}: test MSE {b['test_mse']:.2f}", (b["best_k"], b["test_mse"]),
                xytext=(14, 10.5), color=TXT, fontsize=10, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100], ["1", "2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("k (log scale)")
    ax.set_ylabel("MSE")
    ax.set_ylim(0, 24)
    ax.legend(labelcolor=TXT, loc="lower right", fontsize=9.5)
    ax.set_title("KNN regression across k (70 / 30 holdout)")
    save("knn_curve.png")

    # 5. the AT x V interaction from the enhanced model
    tr, _ = split(df)
    m = fit_ols(h["enhanced"]["formula"], tr)
    import pandas as pd
    at = np.linspace(df.AT.quantile(0.02), df.AT.quantile(0.98), 100)
    fig, ax = plt.subplots(figsize=(10, 4))
    for q, col in [(0.1, AQUA), (0.5, YELLOW), (0.9, ORANGE)]:
        v = df.V.quantile(q)
        grid = pd.DataFrame({"AT": at, "V": v, "AP": df.AP.median(), "RH": df.RH.median()})
        ax.plot(at, m.predict(grid), color=col, label=f"exhaust vacuum = {v:.0f} cm Hg ({int(q*100)}th pct.)")
    ax.set_xlabel("ambient temperature (°C)")
    ax.set_ylabel("predicted output (MW)")
    ax.legend(labelcolor=TXT, fontsize=9.5)
    ax.set_title("Temperature's effect depends on exhaust vacuum (the AT × V interaction)")
    save("interaction.png")


if __name__ == "__main__":
    main()
