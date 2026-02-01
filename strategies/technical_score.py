"""
strategies/technical_score.py
Normalise four technical indicators into [0, 1] scores, then combine with ML probability.
"""
import numpy as np
import pandas as pd


def _score_rsi(rsi: pd.Series) -> pd.Series:
    """30-70 maps linearly 0.3→0.9; below 30 stays ~0.3; above 70 decays."""
    rsi = rsi.fillna(50)
    score = pd.Series(0.5, index=rsi.index)
    lo = rsi < 30
    mid = (rsi >= 30) & (rsi <= 70)
    hi = rsi > 70
    score[lo] = 0.3
    score[mid] = 0.3 + (rsi[mid] - 30) / 40 * 0.6
    score[hi] = 0.9 - (rsi[hi] - 70) * 0.02
    return score.clip(0, 1)


def _score_macd(macd_hist: pd.Series) -> pd.Series:
    """Sigmoid on z-scored MACD histogram."""
    m = macd_hist.fillna(0)
    z = (m - m.mean()) / (m.std() + 1e-8)
    return (1 / (1 + np.exp(-z)))


def _score_adx(adx: pd.Series) -> pd.Series:
    """Piecewise: <20 weak, 20-40 developing, >40 strong."""
    adx = adx.fillna(20)
    score = pd.Series(0.3, index=adx.index)
    w = adx < 20
    m = (adx >= 20) & (adx <= 40)
    s = adx > 40
    score[w] = adx[w] / 20 * 0.3
    score[m] = 0.3 + (adx[m] - 20) / 20 * 0.4
    score[s] = 0.7 + ((adx[s] - 40) / 60 * 0.3)
    return score.clip(0, 1)


def _score_bb(bb_pct: pd.Series) -> pd.Series:
    """0→0.3 at lower band, 0.5→0.6, 1→0.9 at upper band; outside bands penalised."""
    bb = bb_pct.fillna(0.5)
    score = pd.Series(0.5, index=bb.index)
    below = bb < 0
    lo = (bb >= 0) & (bb < 0.5)
    hi = (bb >= 0.5) & (bb <= 1)
    above = bb > 1
    score[below] = 0.3
    score[lo] = 0.3 + bb[lo] / 0.5 * 0.3
    score[hi] = 0.6 + (bb[hi] - 0.5) / 0.5 * 0.3
    score[above] = 0.9 - (bb[above] - 1) * 0.2
    return score.clip(0, 1)


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds TechScore and FinalScore columns.
    FinalScore = 0.6 * ML_Probability + 0.4 * TechScore
    """
    df = df.copy()
    df["RSI_Score"] = _score_rsi(df["RSI_14"])
    df["MACD_Score"] = _score_macd(df["MACD_Hist"])
    df["ADX_Score"] = _score_adx(df["ADX_14"])
    df["BB_Score"] = _score_bb(df["BB_PercentB"])

    df["TechScore"] = (
        0.30 * df["RSI_Score"]
        + 0.30 * df["MACD_Score"]
        + 0.25 * df["ADX_Score"]
        + 0.15 * df["BB_Score"]
    ).clip(0, 1)

    ml = df["ML_Probability"].fillna(0.5)
    df["FinalScore"] = (0.6 * ml + 0.4 * df["TechScore"]).clip(0, 1)
    return df
