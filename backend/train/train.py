"""Training pipeline: EDA -> baseline vs tuned XGBoost -> stratified k-fold CV ->
isotonic calibration -> SHAP TreeExplainer -> serialized artifacts.
Run:  cd backend && python train/train.py"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss,  # noqa: E402
                             classification_report, mean_absolute_error, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,  # noqa: E402
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.models.schemas import canonical_region, canonical_regime  # noqa: E402
from train.make_dataset import generate_dataset  # noqa: E402

BACKEND = Path(__file__).resolve().parents[1]
DATA_DIR, ART_DIR = BACKEND / "data", BACKEND / "artifacts"
NUM_FEATURES = ["lead_time", "rainfall_anomaly_mm", "temp_anomaly_c", "wind_shear_anomaly_kt",
                "geopotential_500hpa_err", "historical_regime_bust_rate"]
CAT_FEATURES = ["region", "weather_regime"]
RANGES = {"lead_time": (1, 10), "rainfall_anomaly_mm": (0, 250), "temp_anomaly_c": (-8, 10),
          "wind_shear_anomaly_kt": (-35, 45), "geopotential_500hpa_err": (-120, 160),
          "historical_regime_bust_rate": (0, 1), "bust_probability": (0, 1)}
PARAM_DIST = {
    "xgb__n_estimators": [200, 300, 400, 600], "xgb__max_depth": [3, 4, 5, 6],
    "xgb__learning_rate": [0.03, 0.05, 0.08, 0.10], "xgb__subsample": [0.7, 0.8, 0.9, 1.0],
    "xgb__colsample_bytree": [0.7, 0.8, 0.9, 1.0], "xgb__min_child_weight": [1, 3, 5, 8],
    "xgb__gamma": [0.0, 0.1, 0.3], "xgb__reg_lambda": [1.0, 2.0, 5.0],
}


# --------------------------------------------------------------------- data
def load_data() -> pd.DataFrame:
    csv = DATA_DIR / "nwp_eval_dataset.csv"
    if not csv.exists():
        print("!! nwp_eval_dataset.csv not found — generating a SYNTHETIC stand-in.")
        print("!! Replace with the provided evaluation CSV and re-run for real numbers.")
        generate_dataset(csv, n=16000, seed=42)
    df = pd.read_csv(csv)
    missing = set(NUM_FEATURES + CAT_FEATURES + ["is_bust", "bust_probability"]) - set(df.columns)
    if missing:
        raise SystemExit(f"Dataset missing required columns: {missing}")
    df["region"] = df["region"].map(canonical_region)
    df["weather_regime"] = df["weather_regime"].map(canonical_regime)
    for col, (lo, hi) in RANGES.items():
        bad = int(((df[col] < lo) | (df[col] > hi)).sum())
        if bad:
            print(f"  [range] {col}: {bad} rows outside [{lo}, {hi}] -> clipped")
        df[col] = df[col].clip(lo, hi)
    return df.dropna(subset=NUM_FEATURES + CAT_FEATURES + ["is_bust"])


# ---------------------------------------------------------------------- EDA
def _bar(series, title, ylabel, path, rot=0):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(i) for i in series.index], series.values, color="#0ea5e9")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=rot, ha="right" if rot else "center")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_eda(df: pd.DataFrame, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n[EDA] rows={len(df)}  bust rate={df.is_bust.mean():.3f}")
    _bar(df.groupby('lead_time')['is_bust'].mean(), "Bust rate by lead time",
         "bust rate", out / "bust_by_lead.png")
    _bar(df.groupby('weather_regime')['is_bust'].mean().sort_values(ascending=False),
         "Bust rate by weather regime", "bust rate", out / "bust_by_regime.png", rot=25)
    corr = df[NUM_FEATURES + ["is_bust", "bust_probability"]].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Correlation matrix")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "correlation.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------- train
def make_pre():
    return ColumnTransformer([
        ("num", "passthrough", NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ])


def main():
    t_start = datetime.now(timezone.utc)
    df = load_data()
    run_eda(df, ART_DIR / "eda")

    X, y = df[NUM_FEATURES + CAT_FEATURES], df["is_bust"].astype(int)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)
    print(f"[split] train={len(X_tr)}  calibration={len(X_cal)}  test={len(X_te)}")

    # ---- baseline -------------------------------------------------------
    base = Pipeline([("pre", make_pre()), ("clf", LogisticRegression(max_iter=2000))])
    base.fit(X_tr, y_tr)

    # ---- tuned XGBoost --------------------------------------------------
    pipe = Pipeline([("pre", make_pre()), ("xgb", XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        random_state=42, n_jobs=-1))])
    search = RandomizedSearchCV(pipe, PARAM_DIST, n_iter=18, scoring="average_precision",
                                cv=StratifiedKFold(3, shuffle=True, random_state=42),
                                n_jobs=-1, random_state=42, verbose=1, refit=True)
    search.fit(X_tr, y_tr)
    best = search.best_estimator_

    # ---- stratified 5-fold CV on the training partition ------------------
    cv = StratifiedKFold(5, shuffle=True, random_state=7)
    pr_cv = cross_val_score(best, X_tr, y_tr, cv=cv, scoring="average_precision", n_jobs=-1)
    roc_cv = cross_val_score(best, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"[cv] PR-AUC {pr_cv.mean():.4f} ± {pr_cv.std():.4f} | ROC-AUC {roc_cv.mean():.4f} ± {roc_cv.std():.4f}")

    # ---- isotonic calibration on a dedicated hold-out --------------------
    raw_cal = best.predict_proba(X_cal)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_cal, y_cal)
    raw_te = best.predict_proba(X_te)[:, 1]
    p_te = np.clip(iso.predict(raw_te), 0.0, 1.0)

    def report(name, p):
        return {"model": name, "pr_auc": round(average_precision_score(y_te, p), 4),
                "roc_auc": round(roc_auc_score(y_te, p), 4), "brier": round(brier_score_loss(y_te, p), 4)}

    rows = [report("LogisticRegression (baseline)", base.predict_proba(X_te)[:, 1]),
            report("XGBoost (raw)", raw_te),
            report("XGBoost (isotonic-calibrated)", p_te)]
    print("\n================ PERFORMANCE REPORT (test set) ================")
    print(f"{'Model':<34}{'PR-AUC':>8}{'ROC-AUC':>9}{'Brier':>8}")
    for r in rows:
        print(f"{r['model']:<34}{r['pr_auc']:>8.4f}{r['roc_auc']:>9.4f}{r['brier']:>8.4f}")
    print(f"Stratified 5-fold CV: PR-AUC {pr_cv.mean():.4f} ± {pr_cv.std():.4f} | "
          f"ROC-AUC {roc_cv.mean():.4f} ± {roc_cv.std():.4f}")
    mae = mean_absolute_error(df.loc[X_te.index, "bust_probability"], p_te)
    print(f"Calibrated-vs-dataset bust_probability MAE: {mae:.4f}")
    print(f"Gates: PR-AUC >= 0.85 -> {'PASS' if rows[2]['pr_auc'] >= 0.85 else 'CHECK'} | "
          f"Brier <= 0.12 -> {'PASS' if rows[2]['brier'] <= 0.12 else 'CHECK'}")
    print("=================================================================\n")
    print(classification_report(y_te, (p_te >= 0.5).astype(int), digits=3))

    # reliability curve
    pt, pp = calibration_curve(y_te, p_te, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(pp, pt, "s-", color="#0ea5e9", label="XGBoost (calibrated)")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed bust frequency")
    ax.set_title("Reliability curve (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ART_DIR / "eda" / "reliability.png", dpi=120)
    plt.close(fig)

    # ---- SHAP explainer ---------------------------------------------------
    pre = best.named_steps["pre"]
    xgb = best.named_steps["xgb"]
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    bg = X_tr.sample(n=min(200, len(X_tr)), random_state=42)
    Xbg = pd.DataFrame(pre.transform(bg), columns=names)
    explainer = shap.TreeExplainer(xgb, data=Xbg, model_output="probability",
                                   feature_perturbation="interventional")
    sv = np.asarray(explainer.shap_values(Xbg.iloc[:5]))
    err = float(np.abs(explainer.expected_value + sv.reshape(len(sv), -1).sum(axis=1)
                       - xgb.predict_proba(Xbg.iloc[:5])[:, 1]).max())
    print(f"[shap] additivity check max|err| = {err:.6f} (probability space)")

    metrics = {
        "n_rows": int(len(df)), "bust_rate": float(y.mean()),
        "baseline_logreg": rows[0], "xgb_raw": rows[1], "xgb_calibrated": rows[2],
        "cv_pr_auc_mean": float(pr_cv.mean()), "cv_pr_auc_std": float(pr_cv.std()),
        "cv_roc_auc_mean": float(roc_cv.mean()), "cv_roc_auc_std": float(roc_cv.std()),
        "bust_probability_mae": round(float(mae), 4),
        "best_params": {k.replace("xgb__", ""): v for k, v in search.best_params_.items()},
    }
    ART_DIR.mkdir(exist_ok=True)
    joblib.dump({"version": "1.0.0", "created_at": t_start.isoformat(),
                 "preprocessor": pre, "model": xgb, "calibrator": iso,
                 "feature_names": names, "num_features": NUM_FEATURES, "cat_features": CAT_FEATURES,
                 "regions": sorted(df["region"].unique()), "regimes": sorted(df["weather_regime"].unique()),
                 "metrics": metrics},
                ART_DIR / "bust_model.joblib", compress=3)
    joblib.dump({"explainer": explainer, "feature_names": names, "background": Xbg},
                ART_DIR / "explainer.joblib", compress=3)
    (ART_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    for f in ("bust_model.joblib", "explainer.joblib"):
        print(f"[artifacts] {f}: {(ART_DIR / f).stat().st_size / 1e6:.2f} MB")
    print("[done] artifacts ready in backend/artifacts/")


if __name__ == "__main__":
    main()