"""Advanced portfolio analytics.

NEW module — these calculations didn't exist in the original app.

Provides:
  * `compute_position_risk_scores`  - composite 0-100 risk score per holding
  * `find_tlh_candidates`           - tax-loss harvesting opportunities
  * `benchmark_comparison`          - portfolio vs benchmark cumulative return
  * `sector_concentration_hhi`      - sector-level HHI concentration index
  * `correlation_clusters`          - groups of highly-correlated positions
  * `top_holding_smart_money`       - run smart-money rules on top holdings
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-position risk score
# ---------------------------------------------------------------------------
def compute_position_risk_scores(summary: pd.DataFrame) -> pd.DataFrame:
    """Composite 0-100 risk score per position.

    Higher = riskier. Built from:
      * Concentration   (Portfolio Weight)            -> up to 30 pts
      * Drawdown        (Drawdown from 52W High %)    -> up to 25 pts
      * Trend           (Bearish/Neutral/Bullish)     -> up to 20 pts
      * Beta            (>1 adds risk)                -> up to 15 pts
      * Distance from 200 MA (large gap up = stretched, large gap down = breakdown)
                                                     -> up to 10 pts
    """
    if summary.empty:
        return summary.assign(**{"Risk Score": np.nan, "Risk Tier": "N/A"})

    df = summary.copy()

    # Concentration sub-score
    weight = df.get("Portfolio Weight", pd.Series(0, index=df.index)).fillna(0).clip(lower=0)
    concentration = np.clip(weight * 200, 0, 30)  # 15% weight -> 30 pts

    # Drawdown sub-score
    dd = df.get("Drawdown from 52W High %", pd.Series(np.nan, index=df.index)).fillna(0)
    drawdown_pts = np.clip(-dd * 100, 0, 25)  # -25% dd -> 25 pts

    # Trend sub-score
    trend = df.get("Trend Rating", pd.Series("", index=df.index)).fillna("")
    trend_pts = trend.map({"Bearish": 20, "Neutral": 10, "Bullish": 0}).fillna(10)

    # Beta sub-score
    beta = df.get("Beta", pd.Series(np.nan, index=df.index)).fillna(1.0)
    beta_pts = np.clip((beta - 1.0) * 10, 0, 15)  # beta=2.5 -> 15 pts

    # Distance from 200 MA — stretched OR broken-down both add risk
    dist = df.get("Distance from 200 MA %", pd.Series(np.nan, index=df.index)).fillna(0)
    stretch_pts = np.clip(np.abs(dist) * 50, 0, 10)  # 20% gap either way -> 10 pts

    score = concentration + drawdown_pts + trend_pts + beta_pts + stretch_pts
    score = score.clip(0, 100).round(1)

    df["Risk Score"] = score
    df["Risk Tier"] = pd.cut(
        df["Risk Score"],
        bins=[-0.01, 30, 55, 75, 100.01],
        labels=["Low", "Moderate", "Elevated", "High"],
    ).astype(str)
    return df


# ---------------------------------------------------------------------------
# Tax-loss harvesting
# ---------------------------------------------------------------------------
def find_tlh_candidates(
    lots: pd.DataFrame,
    *,
    min_loss_dollars: float = 250.0,
    min_loss_pct: float = -0.05,
) -> pd.DataFrame:
    """Find lots that are sitting on harvestable unrealized losses.

    Surfaces:
      * Long-term vs short-term tax term
      * Days since the lot was opened (proxy for wash-sale risk window)
      * Loss in $ and %, plus a flag for "consider TLH"

    The IRS wash-sale rule prohibits buying a substantially identical security
    within 30 days before OR after the sale. This function flags the loss but
    doesn't enforce wash-sale logic — that's the user's job.
    """
    if lots.empty:
        return pd.DataFrame(
            columns=["Symbol", "Description", "Qty", "Fill Price", "Closing Time",
                     "Market Value", "Total Cost", "Unrealized P&L $",
                     "Unrealized P&L %", "Tax Term", "Days Held", "TLH Candidate"]
        )

    df = lots.copy()
    df["Unrealized P&L %"] = np.where(
        df["Total Cost"].fillna(0) != 0,
        df["Unrealized P&L $"] / df["Total Cost"],
        np.nan,
    )

    try:
        df["Days Held"] = (
            pd.Timestamp.today().normalize() - pd.to_datetime(df["Closing Time"], errors="coerce")
        ).dt.days
    except Exception:
        df["Days Held"] = np.nan

    candidate = (
        (df["Unrealized P&L $"].fillna(0) <= -min_loss_dollars)
        & (df["Unrealized P&L %"].fillna(0) <= min_loss_pct)
    )
    df["TLH Candidate"] = np.where(candidate, "✓", "")

    cols = ["Symbol", "Description", "Qty", "Fill Price", "Closing Time",
            "Market Value", "Total Cost", "Unrealized P&L $",
            "Unrealized P&L %", "Tax Term", "Days Held", "TLH Candidate"]
    out = df[[c for c in cols if c in df.columns]].copy()
    out = out.sort_values(["TLH Candidate", "Unrealized P&L $"], ascending=[False, True])
    return out


def tlh_summary(candidates: pd.DataFrame) -> dict:
    """Roll-up stats for the TLH candidates table."""
    if candidates.empty or "TLH Candidate" not in candidates.columns:
        return {"count": 0, "total_loss": 0.0, "lt_loss": 0.0, "st_loss": 0.0}
    flagged = candidates[candidates["TLH Candidate"] == "✓"]
    total = float(flagged["Unrealized P&L $"].sum())
    lt = float(flagged.loc[flagged["Tax Term"].astype(str).str.contains("Long", case=False, na=False), "Unrealized P&L $"].sum())
    st = total - lt
    return {"count": int(len(flagged)), "total_loss": total, "lt_loss": lt, "st_loss": st}


# ---------------------------------------------------------------------------
# Benchmark comparison
# ---------------------------------------------------------------------------
def benchmark_comparison(
    returns: pd.DataFrame,
    weights: pd.Series,
    benchmark: str = "SPY",
) -> pd.DataFrame:
    """Build a cumulative-return DataFrame comparing the portfolio vs a benchmark.

    Returns a DataFrame indexed by date with columns:
      Portfolio (weighted), <benchmark>, Excess Return
    """
    if returns.empty:
        return pd.DataFrame()
    cols = [c for c in returns.columns if c != benchmark]
    if not cols:
        return pd.DataFrame()
    w = weights.reindex(cols).fillna(0)
    if w.sum() <= 0:
        return pd.DataFrame()
    port_ret = returns[cols].mul(w, axis=1).sum(axis=1)
    port_cum = (1 + port_ret.fillna(0)).cumprod() - 1
    out = pd.DataFrame({"Portfolio": port_cum})
    if benchmark in returns.columns:
        bench_cum = (1 + returns[benchmark].fillna(0)).cumprod() - 1
        out[benchmark] = bench_cum
        out["Excess Return"] = out["Portfolio"] - out[benchmark]
    return out


def benchmark_stats(
    returns: pd.DataFrame,
    weights: pd.Series,
    benchmark: str = "SPY",
    risk_free: float = 0.045,
) -> dict:
    """Compact stats vs benchmark: alpha, beta, info ratio, tracking error."""
    if returns.empty or benchmark not in returns.columns:
        return {}
    cols = [c for c in returns.columns if c != benchmark]
    w = weights.reindex(cols).fillna(0)
    if w.sum() <= 0:
        return {}
    port_ret = returns[cols].mul(w, axis=1).sum(axis=1).dropna()
    bench_ret = returns[benchmark].reindex(port_ret.index).dropna()
    aligned = pd.concat([port_ret, bench_ret], axis=1, keys=["p", "b"]).dropna()
    if len(aligned) < 30:
        return {}

    cov = aligned.cov().iloc[0, 1]
    var = aligned["b"].var()
    beta = float(cov / var) if var else np.nan

    ann_port = float(aligned["p"].mean() * 252)
    ann_bench = float(aligned["b"].mean() * 252)
    alpha = ann_port - (risk_free + beta * (ann_bench - risk_free))

    excess = aligned["p"] - aligned["b"]
    tracking_error = float(excess.std() * np.sqrt(252))
    info_ratio = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() else np.nan

    return {
        "Portfolio CAGR-ish": ann_port,
        f"{benchmark} CAGR-ish": ann_bench,
        "Alpha (Jensen)": alpha,
        "Beta vs benchmark": beta,
        "Tracking error": tracking_error,
        "Information ratio": info_ratio,
    }


# ---------------------------------------------------------------------------
# Sector concentration
# ---------------------------------------------------------------------------
def sector_concentration_hhi(summary: pd.DataFrame) -> pd.DataFrame:
    """Per-sector aggregated weight + an overall HHI value."""
    if "Sector" not in summary.columns:
        return pd.DataFrame()
    df = summary.dropna(subset=["Sector"]).copy()
    df = df[df["Sector"].astype(str).str.strip().ne("")]
    if df.empty:
        return pd.DataFrame()
    grp = df.groupby("Sector", as_index=False).agg(
        **{"Market Value": ("Market Value", "sum"),
           "Portfolio Weight": ("Portfolio Weight", "sum"),
           "Positions": ("Symbol", "nunique")}
    )
    grp = grp.sort_values("Portfolio Weight", ascending=False)
    grp["HHI Contribution"] = grp["Portfolio Weight"] ** 2
    return grp


# ---------------------------------------------------------------------------
# Correlation clusters
# ---------------------------------------------------------------------------
def correlation_clusters(
    returns: pd.DataFrame,
    threshold: float = 0.8,
) -> pd.DataFrame:
    """Pairs of holdings with correlation above `threshold`."""
    if returns.empty or returns.shape[1] < 2:
        return pd.DataFrame(columns=["Symbol A", "Symbol B", "Correlation"])
    corr = returns.corr()
    pairs: list[dict] = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = corr.iloc[i, j]
            if pd.notna(c) and c >= threshold:
                pairs.append({"Symbol A": cols[i], "Symbol B": cols[j], "Correlation": round(float(c), 3)})
    return pd.DataFrame(pairs).sort_values("Correlation", ascending=False) if pairs else pd.DataFrame(columns=["Symbol A", "Symbol B", "Correlation"])


# ---------------------------------------------------------------------------
# Smart-money batch analyzer
# ---------------------------------------------------------------------------
def auto_smart_money_table(summary: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Run lightweight smart-money signals on the top N holdings.

    This is a SUMMARY table only — it pulls the existing technical indicators
    already merged onto `summary` and converts them into a smart-money read.
    Full chart analysis is still done one ticker at a time in the SM tab.
    """
    if summary.empty:
        return pd.DataFrame()
    cols_needed = ["Trend Rating", "Distance from 200 MA %", "Drawdown from 52W High %"]
    if not all(c in summary.columns for c in cols_needed):
        return pd.DataFrame()

    top = summary.head(top_n).copy()

    def _read(row: pd.Series) -> str:
        trend = str(row.get("Trend Rating", ""))
        dist = row.get("Distance from 200 MA %", np.nan)
        dd = row.get("Drawdown from 52W High %", np.nan)
        if trend == "Bullish" and pd.notna(dist) and dist < 0.15:
            return "Watch for continuation longs at SND retest"
        if trend == "Bullish" and pd.notna(dist) and dist >= 0.15:
            return "Extended from 200 MA — wait for pullback to MB"
        if trend == "Bearish":
            return "Look for distribution / short setups at supply"
        if trend == "Neutral" and pd.notna(dd) and dd < -0.20:
            return "Possible accumulation — wait for spring confirmation"
        return "No clean smart-money setup yet — sit out"

    top["Smart Money Read"] = top.apply(_read, axis=1)
    cols = ["Symbol", "Description", "Market Value", "Portfolio Weight",
            "Trend Rating", "Distance from 200 MA %",
            "Drawdown from 52W High %", "Smart Money Read"]
    return top[[c for c in cols if c in top.columns]]
