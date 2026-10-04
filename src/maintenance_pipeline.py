"""
Predictive Maintenance - NASA C-MAPSS turbofan (FD001)
Run from the maintenance/ folder:  python src/maintenance_pipeline.py

Goal: for every engine at every cycle, predict whether it will fail within the
next WINDOW cycles, so maintenance can be scheduled before the failure.

Data: put the unzipped C-MAPSS files anywhere under maintenance/data/
(the script searches for train_FD001.txt).
"""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save plots to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
WINDOW = 30           # "failure within the next 30 cycles" = positive label
TARGET_RECALL = 0.80  # alert threshold is chosen to catch >= 80% of at-risk rows

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


# ----------------------------------------------------------------------------
# 1. Load data and build labels
# ----------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    hits = list(DATA_DIR.rglob("train_FD001.txt"))
    if not hits:
        raise FileNotFoundError(
            "train_FD001.txt not found under maintenance/data/. "
            "Unzip the C-MAPSS download there."
        )
    df = pd.read_csv(hits[0], sep=r"\s+", header=None, names=COLS)
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)

    # Every training engine runs until it fails, so remaining life is known
    df["rul"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    df["label"] = (df["rul"] <= WINDOW).astype(int)
    return df


# ----------------------------------------------------------------------------
# 2. Features: rolling statistics and lags (past + current cycle only -> no leakage)
# ----------------------------------------------------------------------------
def pick_sensors(df: pd.DataFrame) -> list:
    """Drop sensors / settings that never change (they carry no information)."""
    cand = [c for c in df.columns if c.startswith("s") or c.startswith("op")]
    return [c for c in cand if df[c].std() > 1e-6]


def make_features(df: pd.DataFrame, sensors: list) -> pd.DataFrame:
    g = df.groupby("unit")
    feats = {"cycle": df["cycle"]}  # operating hours are known in real life too
    for c in sensors:
        s = g[c]
        feats[c] = df[c]
        # rolling windows only look backwards, so no future data is used
        feats[f"{c}_mean5"] = s.transform(lambda x: x.rolling(5, min_periods=1).mean())
        feats[f"{c}_std5"] = s.transform(lambda x: x.rolling(5, min_periods=2).std())
        feats[f"{c}_mean15"] = s.transform(lambda x: x.rolling(15, min_periods=1).mean())
        feats[f"{c}_change5"] = df[c] - s.shift(5)  # trend vs 5 cycles ago
    X = pd.DataFrame(feats)
    return X.fillna(0.0)


# ----------------------------------------------------------------------------
# 3. Split by ENGINE (never random rows, or neighbouring cycles leak across sets)
# ----------------------------------------------------------------------------
def split_units(units: np.ndarray):
    units = np.array(sorted(units))
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(units)
    n = len(units)
    return units[: int(0.6 * n)], units[int(0.6 * n): int(0.8 * n)], units[int(0.8 * n):]


# ----------------------------------------------------------------------------
# 4. Evaluation helpers
# ----------------------------------------------------------------------------
def pick_threshold(y_true, proba, target_recall=TARGET_RECALL) -> float:
    """Highest-precision threshold that still reaches the target recall (on validation)."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    ok = np.where(rec[:-1] >= target_recall)[0]
    if len(ok) == 0:
        return 0.5
    best = ok[np.argmax(prec[:-1][ok])]
    return float(thr[best])


def row_metrics(y_true, proba, threshold) -> dict:
    pred = proba >= threshold
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "false_alarms": int(((pred == 1) & (y_true == 0)).sum()),
    }


def engine_report(test: pd.DataFrame, threshold: float) -> dict:
    """Maintenance-team view: were engines warned in time, and how often were alarms false?"""
    caught, early_alarm, lead = 0, 0, []
    for _, e in test.groupby("unit"):
        alert = e["risk"] >= threshold
        in_window = e["label"] == 1
        if (alert & in_window).any():
            caught += 1
            lead.append(e.loc[alert & in_window, "rul"].max())  # cycles of warning
        if (alert & ~in_window).any():
            early_alarm += 1
    n = test["unit"].nunique()
    false_rows = int(((test["risk"] >= threshold) & (test["label"] == 0)).sum())
    return {
        "test_engines": n,
        "engines_warned_in_time_pct": round(100 * caught / n, 1),
        "median_warning_lead_cycles": float(np.median(lead)) if lead else 0.0,
        "engines_with_false_alarm_pct": round(100 * early_alarm / n, 1),
        "false_alarm_cycles_per_engine": round(false_rows / n, 1),
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    df = load_data()
    sensors = pick_sensors(df)
    X = make_features(df, sensors)
    y = df["label"]
    print(f"Engines: {df['unit'].nunique()} | rows: {len(df)} | "
          f"positive rate (fails within {WINDOW} cycles): {y.mean():.1%}")
    print(f"Sensors used ({len(sensors)}): {sensors}")

    tr_u, va_u, te_u = split_units(df["unit"].unique())
    m_tr, m_va, m_te = (df["unit"].isin(u) for u in (tr_u, va_u, te_u))
    X_tr, y_tr = X[m_tr], y[m_tr]
    X_va, y_va = X[m_va], y[m_va]
    X_te, y_te = X[m_te], y[m_te]
    print(f"Engines -> train {len(tr_u)}, validation {len(va_u)}, test {len(te_u)}")

    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()  # handles the rare failures
    models = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced"),
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="logloss", random_state=RANDOM_STATE,
        ),
    }

    # Compare models on VALIDATION engines
    rows, thresholds = {}, {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_va)[:, 1]
        thresholds[name] = pick_threshold(y_va, p)
        rows[name] = row_metrics(y_va, p, thresholds[name])
        rows[name]["threshold"] = thresholds[name]
    comparison = pd.DataFrame(rows).T.round(3)
    print(f"\nValidation comparison (threshold set for recall >= {TARGET_RECALL:.0%}):")
    print(comparison)
    comparison.to_csv(OUT_DIR / "model_comparison.csv")

    # Best model by PR-AUC (right metric for rare events), final numbers on TEST engines
    best = comparison["pr_auc"].idxmax()
    model, thr = models[best], thresholds[best]
    test = df.loc[m_te, ["unit", "cycle", "rul", "label"]].copy()
    test["risk"] = model.predict_proba(X_te)[:, 1]

    test_metrics = row_metrics(y_te, test["risk"].values, thr)
    eng = engine_report(test, thr)
    print(f"\nBest model: {best} | alert threshold: {thr:.3f}")
    print("Held-out TEST (row level):", {k: round(v, 3) for k, v in test_metrics.items()})
    print("Held-out TEST (engine level):", eng)
    pd.DataFrame([{**test_metrics, **eng, "model": best, "threshold": thr}]).to_csv(
        OUT_DIR / "test_metrics.csv", index=False
    )

    test.round(4).to_csv(OUT_DIR / "test_predictions.csv", index=False)
    joblib.dump(
        {"model": model, "threshold": thr, "sensors": sensors, "window": WINDOW},
        OUT_DIR / "maintenance_model.joblib",
    )

    # ---- Plots -------------------------------------------------------------
    # Precision-recall curves on test
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, m in models.items():
        p = m.predict_proba(X_te)[:, 1]
        pr, rc, _ = precision_recall_curve(y_te, p)
        ax.plot(rc, pr, label=f"{name} (PR-AUC {average_precision_score(y_te, p):.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall (test engines)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pr_curve.png", dpi=150)
    plt.close(fig)

    # Risk over time for 3 test engines
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, u in zip(axes, sorted(te_u)[:3]):
        e = test[test["unit"] == u]
        ax.plot(e["cycle"], e["risk"], label="risk score")
        ax.axhline(thr, color="red", ls="--", label="alert threshold")
        ax.axvspan(e["cycle"].max() - WINDOW, e["cycle"].max(), color="orange", alpha=0.2,
                   label=f"last {WINDOW} cycles")
        ax.set_title(f"Engine {u}")
        ax.set_xlabel("cycle")
    axes[0].set_ylabel("failure risk")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "risk_over_time.png", dpi=150)
    plt.close(fig)

    # Feature importance (XGBoost)
    xgb = models["xgboost"]
    imp = pd.Series(xgb.feature_importances_, index=X.columns).nlargest(15)[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    imp.plot.barh(ax=ax)
    ax.set_title("Top features (XGBoost)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    print("\nSaved metrics, predictions, model and plots to outputs/")


if __name__ == "__main__":
    main()