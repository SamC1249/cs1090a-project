"""Predict log halo mass (M200c) from observable galaxy properties.

Protocol (overfitting guard):
  - One fixed 60/20/20 train/val/test split (seed 109).
  - All models fit on TRAIN only. Hyperparameters and feature-set choices are
    selected on VAL. TEST is touched exactly once at the end, for the baseline
    and the best model, so the reported test numbers are unbiased.

Models: stellar-mass-only linear regression (baseline), ridge regression,
random forest, and histogram gradient boosting, each run on three observable
feature sets.

Usage:
    python src/train_models.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "tng100-1_snap99_centrals.csv"
RESULTS_DIR = REPO_ROOT / "results"

SEED = 109
TARGET = "log_mhalo"

FEATURE_SETS = {
    "baseline (stellar mass)": ["log_mstar"],
    "photometric": [
        "log_mstar", "color_gr", "color_ri", "color_UV", "color_VK",
        "log_size_star", "log_ssfr", "gas_metallicity", "star_metallicity",
    ],
    "photometric + kinematic": [
        "log_mstar", "color_gr", "color_ri", "color_UV", "color_VK",
        "log_size_star", "log_ssfr", "gas_metallicity", "star_metallicity",
        "log_vmax", "log_veldisp",
    ],
}


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["log_vmax"] = np.log10(df["vmax"])
    df["log_veldisp"] = np.log10(np.maximum(df["veldisp"], 1e-2))
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[TARGET] + FEATURE_SETS["photometric + kinematic"]
    )
    # 60/20/20 split, fixed once
    train, rest = train_test_split(df, test_size=0.4, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, random_state=SEED)
    return train, val, test


def metrics(model, X, y):
    pred = model.predict(X)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
    }


def candidate_models():
    """(name, constructor) pairs; each constructor takes a hyperparam value."""
    return {
        "linear": [("ols", lambda _: make_pipeline(StandardScaler(), LinearRegression()), [None])],
        "ridge": [
            ("alpha", lambda a: make_pipeline(StandardScaler(), Ridge(alpha=a)),
             [0.01, 0.1, 1.0, 10.0, 100.0]),
        ],
        "random forest": [
            ("max_depth", lambda d: RandomForestRegressor(
                n_estimators=200, max_depth=d, min_samples_leaf=5,
                n_jobs=-1, random_state=SEED), [8, 12, 16, None]),
        ],
        "grad boosting": [
            ("max_iter", lambda n: HistGradientBoostingRegressor(
                max_iter=n, learning_rate=0.08, max_depth=None,
                min_samples_leaf=20, random_state=SEED), [150, 300, 600]),
        ],
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    train, val, test = load_data()
    print(f"split sizes: train={len(train):,} val={len(val):,} test={len(test):,}")

    rows = []
    fitted = {}
    for fs_name, feats in FEATURE_SETS.items():
        Xtr, ytr = train[feats], train[TARGET]
        Xva, yva = val[feats], val[TARGET]
        for model_name, specs in candidate_models().items():
            # baseline feature set only gets the baseline linear model
            if fs_name.startswith("baseline") and model_name != "linear":
                continue
            best = None
            for param_name, ctor, grid in specs:
                for v in grid:
                    m = ctor(v).fit(Xtr, ytr)
                    tr, va = metrics(m, Xtr, ytr), metrics(m, Xva, yva)
                    if best is None or va["rmse"] < best["val_rmse"]:
                        best = {
                            "features": fs_name, "model": model_name,
                            "hyperparam": f"{param_name}={v}",
                            "train_rmse": tr["rmse"], "train_r2": tr["r2"],
                            "val_rmse": va["rmse"], "val_r2": va["r2"],
                        }
                        fitted[(fs_name, model_name)] = m
            rows.append(best)
            print(f"{fs_name:28s} {best['model']:14s} {best['hyperparam']:14s} "
                  f"train RMSE {best['train_rmse']:.4f}  val RMSE {best['val_rmse']:.4f}  "
                  f"val R2 {best['val_r2']:.4f}")

    res = pd.DataFrame(rows).sort_values("val_rmse")

    # final, single use of the test set: baseline + overall best on val
    base_key = ("baseline (stellar mass)", "linear")
    best_row = res.iloc[0]
    best_key = (best_row["features"], best_row["model"])
    for label, key in [("baseline", base_key), ("best", best_key)]:
        feats = FEATURE_SETS[key[0]]
        te = metrics(fitted[key], test[feats], test[TARGET])
        res.loc[
            (res["features"] == key[0]) & (res["model"] == key[1]),
            ["test_rmse", "test_r2"],
        ] = [te["rmse"], te["r2"]]
        print(f"\nTEST ({label}: {key[1]} on {key[0]}): "
              f"RMSE {te['rmse']:.4f} dex, R2 {te['r2']:.4f}")

    res.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print(f"\nwrote {RESULTS_DIR / 'model_comparison.csv'}")

    # save test-set predictions of baseline + best for plotting
    preds = pd.DataFrame({
        "log_mhalo_true": test[TARGET].to_numpy(),
        "log_mstar": test["log_mstar"].to_numpy(),
        "pred_baseline": fitted[base_key].predict(test[FEATURE_SETS[base_key[0]]]),
        "pred_best": fitted[best_key].predict(test[FEATURE_SETS[best_key[0]]]),
    })
    preds.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    # permutation-free importances if the best model is a forest/boosting
    best_model = fitted[best_key]
    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(
            best_model.feature_importances_, index=FEATURE_SETS[best_key[0]]
        ).sort_values(ascending=False)
        imp.to_csv(RESULTS_DIR / "feature_importances.csv")
        print("\nfeature importances (best model):")
        print(imp.to_string())


if __name__ == "__main__":
    main()
