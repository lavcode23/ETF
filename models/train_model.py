"""
models/train_model.py
Walk-forward XGBoost training with a 36-month rolling window.
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

FEATURE_COLS = [
    "MonthlyReturn", "Momentum_1M", "Momentum_3M", "Momentum_6M", "Momentum_12M",
    "Volatility_1M", "Volatility_3M", "SharpeRatio_1M", "SharpeRatio_3M",
    "RSI_14", "MACD_Hist", "ADX_14", "BB_PercentB", "ATR_Pct",
    "Volume_Ratio", "VolumeSpike", "ADX_Pos", "ADX_Neg", "RelativeStrength",
]


def _usable_features(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLS if c in df.columns]


def walk_forward(monthly: pd.DataFrame, window_months: int = 36, retrain_every: int = 1):
    """
    Returns
    -------
    models_dict : {period_str: {model, scaler, accuracy, auc, feature_importance}}
    history_df  : DataFrame with per-period training stats
    feat_cols   : list of feature column names actually used
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    feat_cols = _usable_features(monthly)
    df = monthly.dropna(subset=feat_cols + ["Target"]).copy()
    df["_ym"] = pd.to_datetime(df.index).to_period("M")
    months = sorted(df["_ym"].unique())

    models_dict = {}
    history = []

    for i in range(window_months, len(months), retrain_every):
        train_start = months[i - window_months]
        train_end = months[i - 1]
        test_month = months[i]

        train = df[df["_ym"].between(train_start, train_end)]
        test = df[df["_ym"] == test_month]

        if len(train) < 40 or len(test) == 0:
            continue

        X_tr, y_tr = train[feat_cols], train["Target"]
        X_te, y_te = test[feat_cols], test["Target"]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="logloss",
            use_label_encoder=False,
        )
        model.fit(X_tr_s, y_tr, verbose=False)

        y_pred = model.predict(X_te_s)
        y_prob = model.predict_proba(X_te_s)[:, 1]

        acc = accuracy_score(y_te, y_pred)
        try:
            auc = roc_auc_score(y_te, y_prob)
        except ValueError:
            auc = float("nan")

        key = str(test_month)
        models_dict[key] = {
            "model": model,
            "scaler": scaler,
            "accuracy": acc,
            "auc": auc,
            "feature_importance": dict(zip(feat_cols, model.feature_importances_)),
        }
        history.append({"test_month": key, "accuracy": acc, "auc": auc,
                        "train_n": len(train), "test_n": len(test)})

        # persist
        joblib.dump({"model": model, "scaler": scaler, "feat_cols": feat_cols},
                    os.path.join(MODEL_DIR, f"model_{key}.joblib"))

    print(f"  walk_forward: trained {len(models_dict)} models")
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(MODEL_DIR, "training_history.csv"), index=False)
    return models_dict, history_df, feat_cols


def feature_importance(models_dict: dict) -> pd.DataFrame:
    """Average feature importance across all trained models."""
    rows = [v["feature_importance"] for v in models_dict.values()]
    if not rows:
        return pd.DataFrame()
    imp = pd.DataFrame(rows).mean().sort_values(ascending=False)
    return pd.DataFrame({"Feature": imp.index, "Importance": imp.values})
