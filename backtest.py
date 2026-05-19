"""Backtesting — simulate the current portfolio over historical periods.

Lets the user ask: "If I'd held these exact weights since 2020, how would
I have done compared to just buying SPY?"
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None  # type: ignore


def _fetch_prices_one(symbol: str, start: str, end: str) -> tuple[str, pd.Series | None]:
    if yf is None:
        return symbol, None
    try:
        hist = yf.Ticker(symbol).history(start=start, end=end, interval="1d", auto_adjust=True)
        if hist is None or hist.empty or "Close" not in hist:
            return symbol, None
        return symbol, hist["Close"]
    except Exception:
        return symbol, None


def backtest_portfolio(
    symbols: Iterable[str],
    weights: pd.Series,
    start: str,
    end: str | None = None,
    benchmark: str = "SPY",
    starting_value: float = 100_000.0,
    rebalance: str = "none",
    max_workers: int = 8,
) -> dict:
    """Simulate the portfolio with given weights from `start` to `end`.

    rebalance: "none" (buy-and-hold), "quarterly", or "annually"
    Returns dict with value curves and computed metrics.
    """
    if yf is None:
        return {}
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    syms = list(weights.index[weights > 0])
    if not syms:
        return {}
    syms_plus_bench = list(set(syms + [benchmark]))

    prices_data: dict[str, pd.Series] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_fetch_prices_one, s, start, end) for s in syms_plus_bench]
        for f in as_completed(futures):
            sym, series = f.result()
            if series is not None:
                prices_data[sym] = series

    if not prices_data or benchmark not in prices_data:
        return {}

    prices = pd.DataFrame(prices_data).dropna(how="all")
    if prices.empty or len(prices) < 30:
        return {}

    bench_prices = prices[benchmark].dropna()
    bench_curve = (bench_prices / bench_prices.iloc[0]) * starting_value

    port_cols = [s for s in syms if s in prices.columns]
    port_prices = prices[port_cols].dropna(how="all").ffill()
    w = weights.reindex(port_cols).fillna(0)
    w = w / w.sum() if w.sum() > 0 else w

    daily_rets = port_prices.pct_change().fillna(0)

    if rebalance == "none":
        # Buy-and-hold: build value as weight × cumulative return
        cum_per_symbol = (1 + daily_rets).cumprod()
        port_curve = (cum_per_symbol * w * starting_value).sum(axis=1)
    else:
        # Periodic rebalancing
        rebal_periods = {"quarterly": "Q", "annually": "A"}
        rebal_freq = rebal_periods.get(rebalance, "Q")
        port_curve = pd.Series(index=daily_rets.index, dtype=float)
        current_value = starting_value
        current_alloc = (w * current_value).to_dict()
        rebal_dates = set(pd.date_range(daily_rets.index[0], daily_rets.index[-1], freq=rebal_freq))

        for date in daily_rets.index:
            ret_row = daily_rets.loc[date]
            for sym in port_cols:
                current_alloc[sym] *= (1 + ret_row[sym])
            current_value = sum(current_alloc.values())
            port_curve.loc[date] = current_value
            if date in rebal_dates:
                current_alloc = {s: w[s] * current_value for s in port_cols}

    port_curve = port_curve.reindex(bench_curve.index, method="ffill")

    # Daily returns of the curves for metrics
    port_daily = port_curve.pct_change().dropna()
    bench_daily = bench_curve.pct_change().dropna()

    metrics = _compute_backtest_metrics(port_daily, bench_daily, starting_value,
                                        port_curve.iloc[-1], bench_curve.iloc[-1])

    return {
        "port_curve": port_curve,
        "bench_curve": bench_curve,
        "metrics": metrics,
        "start_date": port_curve.index[0].strftime("%Y-%m-%d"),
        "end_date": port_curve.index[-1].strftime("%Y-%m-%d"),
        "n_days": int(len(port_curve)),
        "starting_value": starting_value,
        "benchmark": benchmark,
        "rebalance": rebalance,
    }


def _compute_backtest_metrics(
    port_daily: pd.Series,
    bench_daily: pd.Series,
    starting_value: float,
    final_port: float,
    final_bench: float,
) -> dict:
    """Compute the standard performance metrics."""
    n_years = max(len(port_daily) / 252, 0.001)

    port_cagr = (final_port / starting_value) ** (1 / n_years) - 1
    bench_cagr = (final_bench / starting_value) ** (1 / n_years) - 1

    port_vol = port_daily.std() * np.sqrt(252)
    bench_vol = bench_daily.std() * np.sqrt(252)

    port_sharpe = port_cagr / port_vol if port_vol else np.nan
    bench_sharpe = bench_cagr / bench_vol if bench_vol else np.nan

    # Sortino (downside vol)
    port_downside = port_daily[port_daily < 0]
    sortino = (port_cagr / (port_downside.std() * np.sqrt(252))) if len(port_downside) > 5 and port_downside.std() > 0 else np.nan

    # Max drawdown
    port_cum = (1 + port_daily).cumprod()
    port_dd = float((port_cum / port_cum.cummax() - 1).min())
    bench_cum = (1 + bench_daily).cumprod()
    bench_dd = float((bench_cum / bench_cum.cummax() - 1).min())

    # Calmar
    calmar = port_cagr / abs(port_dd) if port_dd else np.nan

    # Win rate
    win_rate = float((port_daily > 0).mean()) if len(port_daily) > 0 else np.nan

    # Beta vs benchmark
    if len(port_daily) > 30 and len(bench_daily) > 30:
        aligned = pd.concat([port_daily, bench_daily], axis=1).dropna()
        if len(aligned) > 30:
            cov = aligned.cov().iloc[0, 1]
            var = aligned.iloc[:, 1].var()
            beta = float(cov / var) if var else np.nan
        else:
            beta = np.nan
    else:
        beta = np.nan

    return {
        "Portfolio CAGR": port_cagr,
        "Benchmark CAGR": bench_cagr,
        "Excess CAGR": port_cagr - bench_cagr,
        "Portfolio Vol": port_vol,
        "Benchmark Vol": bench_vol,
        "Portfolio Sharpe": port_sharpe,
        "Benchmark Sharpe": bench_sharpe,
        "Portfolio Sortino": sortino,
        "Portfolio Max DD": port_dd,
        "Benchmark Max DD": bench_dd,
        "Calmar Ratio": calmar,
        "Win Rate (daily)": win_rate,
        "Beta vs Benchmark": beta,
        "Final Portfolio Value": float(final_port),
        "Final Benchmark Value": float(final_bench),
    }
