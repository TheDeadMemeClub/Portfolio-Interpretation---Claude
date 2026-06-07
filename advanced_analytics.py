"""Advanced analytics — institutional-grade portfolio diagnostics.

Adds the things that separate professional investors from retail:
  * `performance_attribution` - which holdings actually drove portfolio return
  * `monte_carlo_projection`  - bootstrap forward simulation (retirement planning)
  * `historical_stress_test`  - apply 2008, COVID, 2022 etc. to current portfolio
  * `factor_exposure`         - OLS regression vs value/momentum/quality factors
  * `efficient_frontier`      - random-portfolio scatter with current portfolio plotted
  * `portfolio_grade`         - single A+ to F grade on six sub-scores
  * `dividend_growth_analysis`- 5-year div growth + sustainability check
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Performance attribution
# ---------------------------------------------------------------------------
def performance_attribution(
    returns: pd.DataFrame,
    weights: pd.Series,
    period_days: int | None = None,
) -> pd.DataFrame:
    """Per-position contribution to portfolio total return.

    Returns a DataFrame with columns:
      Symbol, Weight, Period Return, Contribution, Contribution %
    Where Contribution = Weight × Period Return (in decimal points).
    """
    if returns.empty:
        return pd.DataFrame()

    if period_days is not None and period_days < len(returns):
        rets = returns.iloc[-period_days:]
    else:
        rets = returns

    period_return = (1 + rets).prod() - 1  # geometric compounded return per symbol
    w = weights.reindex(period_return.index).fillna(0)

    contribution = w * period_return
    total = contribution.sum()

    df = pd.DataFrame({
        "Symbol": period_return.index,
        "Weight": w.values,
        "Period Return": period_return.values,
        "Contribution": contribution.values,
        "Contribution %": (contribution / total if total else 0).values,
    })
    return df.sort_values("Contribution", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Monte Carlo projection
# ---------------------------------------------------------------------------
@dataclass
class MonteCarloResult:
    paths: np.ndarray            # shape (n_sims, days+1)
    percentile_10: np.ndarray    # daily 10th-percentile path
    percentile_50: np.ndarray    # median path
    percentile_90: np.ndarray    # daily 90th-percentile path
    final_values: np.ndarray     # final value distribution
    summary: dict


def monte_carlo_projection(
    returns: pd.DataFrame,
    weights: pd.Series,
    starting_value: float,
    years: int = 10,
    n_sims: int = 5000,
    annual_contribution: float = 0.0,
    rng_seed: int | None = 42,
) -> MonteCarloResult | None:
    """Bootstrap-simulate the portfolio forward.

    Samples (with replacement) from the historical daily portfolio-return series,
    then runs `n_sims` paths over `years * 252` trading days.

    `annual_contribution` is added evenly across the year (annual / 252 per day).
    Returns None if there's not enough history.
    """
    if returns.empty or starting_value <= 0:
        return None
    w = weights.reindex(returns.columns).fillna(0)
    if w.sum() <= 0:
        return None
    port_returns = returns.mul(w, axis=1).sum(axis=1).dropna().values
    if len(port_returns) < 60:
        return None

    rng = np.random.default_rng(rng_seed)
    days = int(years * 252)
    daily_contribution = annual_contribution / 252.0

    # Sample shape (n_sims, days)
    samples = rng.choice(port_returns, size=(n_sims, days), replace=True)
    # Build value paths
    paths = np.empty((n_sims, days + 1), dtype=np.float64)
    paths[:, 0] = starting_value
    for t in range(days):
        paths[:, t + 1] = paths[:, t] * (1 + samples[:, t]) + daily_contribution

    final_vals = paths[:, -1]
    p10_path = np.percentile(paths, 10, axis=0)
    p50_path = np.percentile(paths, 50, axis=0)
    p90_path = np.percentile(paths, 90, axis=0)

    summary = {
        "starting_value": starting_value,
        "years": years,
        "n_sims": n_sims,
        "annual_contribution": annual_contribution,
        "final_p10": float(np.percentile(final_vals, 10)),
        "final_p25": float(np.percentile(final_vals, 25)),
        "final_p50": float(np.percentile(final_vals, 50)),
        "final_p75": float(np.percentile(final_vals, 75)),
        "final_p90": float(np.percentile(final_vals, 90)),
        "final_mean": float(np.mean(final_vals)),
        "prob_double": float(np.mean(final_vals >= starting_value * 2)),
        "prob_triple": float(np.mean(final_vals >= starting_value * 3)),
        "prob_loss": float(np.mean(final_vals < starting_value)),
    }
    return MonteCarloResult(paths, p10_path, p50_path, p90_path, final_vals, summary)


def probability_of_target(result: MonteCarloResult, target_value: float) -> float:
    """Probability of finishing above a specific dollar target."""
    if result is None:
        return float("nan")
    return float(np.mean(result.final_values >= target_value))


# ---------------------------------------------------------------------------
# Historical stress test
# ---------------------------------------------------------------------------
# Named crisis periods with known approximate dates.
# We use these to compute how the user's CURRENT portfolio weights
# would have performed if held during those windows.
CRISIS_PERIODS: list[dict] = [
    {"name": "COVID Crash (2020)", "start": "2020-02-19", "end": "2020-03-23",
     "description": "Pandemic onset: 33 days, S&P -34%"},
    {"name": "Inflation Selloff (2022)", "start": "2022-01-03", "end": "2022-10-12",
     "description": "Fed tightening cycle, S&P -25%"},
    {"name": "Q4 2018 Rate Scare", "start": "2018-09-20", "end": "2018-12-24",
     "description": "Rate hikes + trade war, S&P -19%"},
    {"name": "2015-16 China/Oil", "start": "2015-08-17", "end": "2016-02-11",
     "description": "Yuan devaluation + oil crash, S&P -13%"},
    {"name": "2011 EU Debt Crisis", "start": "2011-04-29", "end": "2011-10-03",
     "description": "Greek crisis + US downgrade, S&P -19%"},
    {"name": "2008 Financial Crisis", "start": "2007-10-09", "end": "2009-03-09",
     "description": "Lehman collapse, S&P -57% peak-to-trough"},
]


def historical_stress_test(
    symbols: Iterable[str],
    weights: pd.Series,
    fetch_history_fn,
    benchmark: str = "SPY",
) -> pd.DataFrame:
    """Apply each named crisis to the CURRENT portfolio weights.

    `fetch_history_fn(symbol, start, end) -> pd.Series` of closing prices.
    """
    syms = [s for s in symbols if weights.get(s, 0) > 0]
    if not syms:
        return pd.DataFrame()

    rows: list[dict] = []
    for crisis in CRISIS_PERIODS:
        start, end = crisis["start"], crisis["end"]
        # Pull aligned price history for each holding
        port_perf, bench_perf = _compute_period_perf(syms, weights, start, end, fetch_history_fn, benchmark)
        if port_perf is None:
            continue
        rows.append({
            "Crisis": crisis["name"],
            "Window": f"{start} to {end}",
            "Description": crisis["description"],
            "Portfolio Return": port_perf,
            f"{benchmark} Return": bench_perf,
            "Excess vs Benchmark": (port_perf - bench_perf) if bench_perf is not None else np.nan,
        })
    return pd.DataFrame(rows)


def _compute_period_perf(symbols, weights, start, end, fetch_fn, benchmark):
    """Helper: compute total return for portfolio + benchmark over a date window."""
    port_return = 0.0
    weight_used = 0.0
    for sym in symbols:
        w = float(weights.get(sym, 0))
        if w <= 0:
            continue
        try:
            prices = fetch_fn(sym, start, end)
            if prices is None or len(prices) < 2:
                continue
            p_start = float(prices.iloc[0])
            p_end = float(prices.iloc[-1])
            if p_start <= 0:
                continue
            sym_return = (p_end / p_start) - 1
            port_return += w * sym_return
            weight_used += w
        except Exception:
            continue
    if weight_used <= 0:
        return None, None
    # Scale to actual weight covered (some symbols may not have history)
    port_return = port_return / weight_used if weight_used else port_return

    bench_return = None
    try:
        bench_prices = fetch_fn(benchmark, start, end)
        if bench_prices is not None and len(bench_prices) >= 2:
            bench_return = (float(bench_prices.iloc[-1]) / float(bench_prices.iloc[0])) - 1
    except Exception:
        pass
    return port_return, bench_return


# ---------------------------------------------------------------------------
# Factor exposure (style analysis)
# ---------------------------------------------------------------------------
# Proxy ETFs for major factors — chosen because they're highly liquid and
# yfinance-friendly. Replace with Fama-French data if you have a feed.
FACTOR_PROXIES: dict[str, str] = {
    "Market":    "SPY",
    "Value":     "VLUE",
    "Momentum":  "MTUM",
    "Quality":   "QUAL",
    "Size":      "SIZE",
    "LowVol":    "USMV",
    "Growth":    "VUG",
}


def factor_exposure(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> dict:
    """OLS regression of portfolio excess returns on factor returns.

    Returns dict with:
      betas:    {factor_name: beta}
      alpha:    annualized intercept
      r2:       R-squared of the regression
      t_stats:  approximate t-stats for each beta
    """
    if portfolio_returns.empty or factor_returns.empty:
        return {}
    aligned = pd.concat([portfolio_returns.rename("port"), factor_returns], axis=1).dropna()
    if len(aligned) < 60:
        return {}

    y = aligned["port"].values
    X_cols = [c for c in factor_returns.columns if c in aligned.columns]
    X = aligned[X_cols].values
    X_with_const = np.column_stack([np.ones(len(X)), X])

    # OLS: beta = (X'X)^-1 X'y
    try:
        coefs, residuals_arr, rank, sv = np.linalg.lstsq(X_with_const, y, rcond=None)
    except np.linalg.LinAlgError:
        return {}

    alpha_daily = coefs[0]
    betas = dict(zip(X_cols, coefs[1:]))

    y_pred = X_with_const @ coefs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0

    # Approximate standard errors and t-stats
    n, k = X_with_const.shape
    if n - k > 0 and ss_res > 0:
        sigma2 = ss_res / (n - k)
        try:
            cov = sigma2 * np.linalg.inv(X_with_const.T @ X_with_const)
            se = np.sqrt(np.diag(cov))
            t_stats = coefs / se
            t_stats_dict = dict(zip(["alpha"] + X_cols, t_stats))
        except np.linalg.LinAlgError:
            t_stats_dict = {}
    else:
        t_stats_dict = {}

    return {
        "betas": betas,
        "alpha": float(alpha_daily * 252),   # annualized
        "alpha_daily": float(alpha_daily),
        "r2": float(r2),
        "t_stats": {k: float(v) for k, v in t_stats_dict.items()},
        "n_obs": int(n),
    }


# ---------------------------------------------------------------------------
# Efficient frontier (random portfolios)
# ---------------------------------------------------------------------------
def efficient_frontier(
    returns: pd.DataFrame,
    current_weights: pd.Series,
    n_portfolios: int = 5000,
    risk_free: float = 0.045,
    rng_seed: int | None = 42,
) -> dict:
    """Random-portfolio Monte Carlo to visualize the efficient frontier.

    Generates `n_portfolios` random weight combinations, computes annualized
    return and vol for each, identifies max-Sharpe and min-variance portfolios,
    and locates the user's current portfolio on the scatter.
    """
    if returns.empty or returns.shape[1] < 2:
        return {}
    symbols = list(returns.columns)
    n = len(symbols)

    # Annualized mean and covariance
    mean_ret = returns.mean() * 252
    cov_mat = returns.cov() * 252

    rng = np.random.default_rng(rng_seed)
    weights = rng.dirichlet(np.ones(n), size=n_portfolios)

    port_returns = weights @ mean_ret.values
    port_vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov_mat.values, weights))
    sharpes = (port_returns - risk_free) / port_vols

    # Locate key portfolios
    max_sharpe_idx = int(np.argmax(sharpes))
    min_vol_idx = int(np.argmin(port_vols))

    # Current portfolio
    cw = current_weights.reindex(symbols).fillna(0).values
    if cw.sum() > 0:
        cw = cw / cw.sum()
        current_ret = float(cw @ mean_ret.values)
        current_vol = float(np.sqrt(cw @ cov_mat.values @ cw))
        current_sharpe = (current_ret - risk_free) / current_vol if current_vol else np.nan
    else:
        current_ret = current_vol = current_sharpe = float("nan")

    def _weights_to_dict(w_array):
        return {sym: float(w) for sym, w in zip(symbols, w_array) if w > 0.005}

    return {
        "scatter_returns": port_returns,
        "scatter_vols": port_vols,
        "scatter_sharpes": sharpes,
        "max_sharpe": {
            "return": float(port_returns[max_sharpe_idx]),
            "vol": float(port_vols[max_sharpe_idx]),
            "sharpe": float(sharpes[max_sharpe_idx]),
            "weights": _weights_to_dict(weights[max_sharpe_idx]),
        },
        "min_vol": {
            "return": float(port_returns[min_vol_idx]),
            "vol": float(port_vols[min_vol_idx]),
            "sharpe": float(sharpes[min_vol_idx]),
            "weights": _weights_to_dict(weights[min_vol_idx]),
        },
        "current": {
            "return": current_ret,
            "vol": current_vol,
            "sharpe": current_sharpe,
            "weights": _weights_to_dict(cw),
        },
    }


# ---------------------------------------------------------------------------
# Portfolio grade scorecard
# ---------------------------------------------------------------------------
def portfolio_grade(
    summary: pd.DataFrame,
    *,
    hhi: float | None = None,
    sharpe: float | None = None,
    max_drawdown: float | None = None,
    cash_weight: float | None = None,
    bullish_weight: float | None = None,
    bearish_weight: float | None = None,
    tlh_unrealized_loss: float | None = None,
    portfolio_value: float | None = None,
) -> dict:
    """Compute six sub-scores (0-100) and aggregate to a letter grade.

    Sub-scores:
      Diversification     - inverse of HHI
      Risk-Adjusted Return - Sharpe ratio bucket
      Drawdown Control    - max drawdown bucket
      Trend Health        - bullish vs bearish weight balance
      Concentration Control - largest position size
      Tax Efficiency      - unused harvestable losses vs portfolio value
    """
    scores: dict[str, float] = {}
    rationale: dict[str, str] = {}

    # 1. Diversification (lower HHI = better; <0.05 = excellent, >0.20 = poor)
    if hhi is not None:
        if hhi <= 0.05:
            scores["Diversification"] = 100
        elif hhi <= 0.10:
            scores["Diversification"] = 85
        elif hhi <= 0.15:
            scores["Diversification"] = 70
        elif hhi <= 0.25:
            scores["Diversification"] = 50
        else:
            scores["Diversification"] = 25
        rationale["Diversification"] = f"HHI = {hhi:.3f} (effective {1/hhi:.1f} positions)"

    # 2. Risk-Adjusted Return (Sharpe)
    if sharpe is not None and not np.isnan(sharpe):
        if sharpe >= 1.5:
            scores["Risk-Adjusted Return"] = 100
        elif sharpe >= 1.0:
            scores["Risk-Adjusted Return"] = 85
        elif sharpe >= 0.5:
            scores["Risk-Adjusted Return"] = 65
        elif sharpe >= 0:
            scores["Risk-Adjusted Return"] = 40
        else:
            scores["Risk-Adjusted Return"] = 15
        rationale["Risk-Adjusted Return"] = f"Sharpe ratio ≈ {sharpe:.2f}"

    # 3. Drawdown Control
    if max_drawdown is not None and not np.isnan(max_drawdown):
        dd = abs(max_drawdown)
        if dd <= 0.05:
            scores["Drawdown Control"] = 100
        elif dd <= 0.10:
            scores["Drawdown Control"] = 85
        elif dd <= 0.15:
            scores["Drawdown Control"] = 65
        elif dd <= 0.25:
            scores["Drawdown Control"] = 40
        else:
            scores["Drawdown Control"] = 20
        rationale["Drawdown Control"] = f"Max DD (1Y) = {max_drawdown:.1%}"

    # 4. Trend Health
    if bullish_weight is not None and bearish_weight is not None:
        net = bullish_weight - bearish_weight
        if net >= 0.50:
            scores["Trend Health"] = 100
        elif net >= 0.30:
            scores["Trend Health"] = 85
        elif net >= 0.10:
            scores["Trend Health"] = 65
        elif net >= -0.10:
            scores["Trend Health"] = 50
        else:
            scores["Trend Health"] = 25
        rationale["Trend Health"] = f"Bull {bullish_weight:.0%} vs Bear {bearish_weight:.0%}"

    # 5. Concentration control (largest position)
    if not summary.empty and "Portfolio Weight" in summary.columns:
        top1 = float(summary["Portfolio Weight"].max())
        if top1 <= 0.06:
            scores["Concentration Control"] = 100
        elif top1 <= 0.10:
            scores["Concentration Control"] = 85
        elif top1 <= 0.15:
            scores["Concentration Control"] = 65
        elif top1 <= 0.25:
            scores["Concentration Control"] = 40
        else:
            scores["Concentration Control"] = 20
        rationale["Concentration Control"] = f"Largest position = {top1:.1%}"

    # 6. Tax efficiency (unused harvestable losses)
    if tlh_unrealized_loss is not None and portfolio_value:
        loss_pct = abs(tlh_unrealized_loss) / portfolio_value
        if loss_pct <= 0.005:
            scores["Tax Efficiency"] = 100
        elif loss_pct <= 0.015:
            scores["Tax Efficiency"] = 80
        elif loss_pct <= 0.03:
            scores["Tax Efficiency"] = 60
        else:
            scores["Tax Efficiency"] = 35
        rationale["Tax Efficiency"] = f"${abs(tlh_unrealized_loss):,.0f} unharvested ({loss_pct:.1%})"

    if not scores:
        return {"grade": "—", "score": 0, "sub_scores": {}, "rationale": {}}

    overall = float(np.mean(list(scores.values())))
    if overall >= 92:
        letter = "A+"
    elif overall >= 85:
        letter = "A"
    elif overall >= 78:
        letter = "A-"
    elif overall >= 72:
        letter = "B+"
    elif overall >= 65:
        letter = "B"
    elif overall >= 58:
        letter = "B-"
    elif overall >= 50:
        letter = "C"
    elif overall >= 40:
        letter = "D"
    else:
        letter = "F"

    return {
        "grade": letter,
        "score": round(overall, 1),
        "sub_scores": {k: round(v, 1) for k, v in scores.items()},
        "rationale": rationale,
    }


# ---------------------------------------------------------------------------
# Dividend growth analysis
# ---------------------------------------------------------------------------
def dividend_growth_analysis(symbols: Iterable[str], dividend_fetch_fn) -> pd.DataFrame:
    """Analyze dividend growth and consistency for each symbol.

    `dividend_fetch_fn(symbol) -> pd.Series` indexed by date, values = dividend amount.

    Returns one row per symbol with:
      5Y Dividend CAGR, Years of Consecutive Growth, TTM Dividend,
      Dividend Consistency (% years with growth), Last Dividend Date.
    """
    rows: list[dict] = []
    for sym in symbols:
        try:
            div = dividend_fetch_fn(sym)
            if div is None or div.empty:
                continue
            div = div.dropna()
            if div.empty:
                continue
            # yfinance dividend indexes are tz-aware (e.g. America/New_York).
            # Strip the tz so comparisons against tz-naive cutoffs don't raise.
            idx = pd.to_datetime(div.index)
            try:
                if getattr(idx, "tz", None) is not None:
                    idx = idx.tz_localize(None)
            except (TypeError, AttributeError):
                idx = idx.tz_localize(None) if hasattr(idx, "tz_localize") else idx
            df = pd.DataFrame({"date": idx, "amount": div.values})
            df["year"] = df["date"].dt.year
            annual = df.groupby("year")["amount"].sum().sort_index()
            if len(annual) < 2:
                continue
            # 5-year CAGR
            cagr_5y = np.nan
            if len(annual) >= 5:
                first, last = annual.iloc[-6], annual.iloc[-1]
                if first > 0:
                    cagr_5y = (last / first) ** (1 / 5) - 1
            elif len(annual) >= 2:
                first, last = annual.iloc[0], annual.iloc[-1]
                if first > 0:
                    yrs = annual.index[-1] - annual.index[0]
                    cagr_5y = (last / first) ** (1 / yrs) - 1 if yrs else np.nan
            # Consecutive growth years
            consec = 0
            for i in range(len(annual) - 1, 0, -1):
                if annual.iloc[i] > annual.iloc[i - 1]:
                    consec += 1
                else:
                    break
            # Consistency (% of annual periods showing growth)
            diffs = annual.diff().dropna()
            consistency = (diffs > 0).mean() if len(diffs) > 0 else np.nan
            # TTM dividend
            cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=365)
            ttm = float(df[df["date"] >= cutoff]["amount"].sum())

            rows.append({
                "Symbol": sym,
                "TTM Dividend": ttm,
                "5Y Dividend CAGR": cagr_5y,
                "Years Consecutive Growth": consec,
                "Growth Consistency": consistency,
                "Last Dividend Date": df["date"].max().strftime("%Y-%m-%d"),
                "Years of History": int(len(annual)),
            })
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("5Y Dividend CAGR", ascending=False, na_position="last")
    return out.reset_index(drop=True)
