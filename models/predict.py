"""
models/predict.py
Stamp ML_Probability onto each (ETF, month) row using the closest trained model.
"""
import pandas as pd
import numpy as np


def predict_all(monthly: pd.DataFrame, models_dict: dict, feat_cols: list[str]) -> pd.DataFrame:
    """
    For every row in *monthly* find the most recent model whose test_month <= that row's month,
    run predict_proba, and store result in ML_Probability.
    """
    df = monthly.copy()
    df["_ym"] = pd.to_datetime(df.index).to_period("M")
    df["ML_Probability"] = np.nan

    sorted_keys = sorted(models_dict.keys())   # period strings sort lexicographically fine

    for idx, row in df.iterrows():
        row_ym = str(df.loc[idx, "_ym"] if isinstance(df.loc[idx, "_ym"], object) else df.loc[idx, "_ym"])
        # find closest model key <= row_ym
        candidates = [k for k in sorted_keys if k <= row_ym]
        if not candidates:
            continue
        key = candidates[-1]
        md = models_dict[key]

        X = row[feat_cols].values.reshape(1, -1)
        X_filled = np.nan_to_num(X, nan=0.0)
        X_s = md["scaler"].transform(X_filled)
        prob = md["model"].predict_proba(X_s)[0, 1]
        df.at[idx, "ML_Probability"] = prob

    df.drop(columns=["_ym"], inplace=True, errors="ignore")
    print(f"  predict_all: filled {df['ML_Probability'].notna().sum()} / {len(df)} rows")
    return df
