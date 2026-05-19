"""Options analysis — covered-call screener + IV analysis.

Pulls options chains via yfinance and turns them into income opportunities
for stock positions the user already owns.
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


# ---------------------------------------------------------------------------
# Single-symbol options chain
# ---------------------------------------------------------------------------
def get_options_chain(symbol: str, expiry: str | None = None) -> dict:
    """Return calls + puts for one expiry (or the nearest one if None).

    Returns dict with keys: spot, expiry, calls (DataFrame), puts (DataFrame).
    """
    if yf is None:
        return {}
    try:
        tk = yf.Ticker(symbol)
        expiries = list(tk.options or [])
        if not expiries:
            return {}
        target_expiry = expiry if expiry in expiries else expiries[0]
        chain = tk.option_chain(target_expiry)
        try:
            spot = float(tk.fast_info.get("last_price", np.nan))
        except Exception:
            spot = np.nan
        return {
            "spot": spot,
            "expiry": target_expiry,
            "expiries": expiries,
            "calls": chain.calls,
            "puts": chain.puts,
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Covered-call screener
# ---------------------------------------------------------------------------
def _covered_call_one(
    symbol: str,
    shares: float,
    target_dte_min: int,
    target_dte_max: int,
    otm_pct_target: float,
) -> list[dict]:
    """Find the best covered-call opportunity for one symbol."""
    if yf is None or shares < 100:
        return []
    try:
        tk = yf.Ticker(symbol)
        expiries = list(tk.options or [])
        if not expiries:
            return []
        try:
            spot = float(tk.fast_info.get("last_price", np.nan))
        except Exception:
            spot = np.nan
        if np.isnan(spot) or spot <= 0:
            return []

        today = pd.Timestamp.today().normalize()
        candidates: list[dict] = []
        target_strike = spot * (1 + otm_pct_target)

        for exp_str in expiries:
            exp = pd.Timestamp(exp_str)
            dte = (exp - today).days
            if dte < target_dte_min or dte > target_dte_max:
                continue
            try:
                chain = tk.option_chain(exp_str)
            except Exception:
                continue
            calls = chain.calls.copy()
            if calls.empty or "strike" not in calls.columns:
                continue
            # Find OTM calls only
            calls = calls[calls["strike"] >= spot].copy()
            if calls.empty:
                continue
            # Pick the strike closest to our target OTM
            calls["distance"] = (calls["strike"] - target_strike).abs()
            best = calls.nsmallest(1, "distance").iloc[0]

            strike = float(best["strike"])
            mid = float(best.get("bid", 0) + best.get("ask", 0)) / 2 if best.get("ask", 0) > 0 else float(best.get("lastPrice", 0))
            if mid <= 0:
                continue

            contracts = int(shares // 100)
            premium_total = mid * 100 * contracts
            if_called_total = (strike - spot) * 100 * contracts + premium_total
            otm_pct = (strike / spot - 1)
            annualized_premium_yield = (mid / spot) * (365 / dte) if dte > 0 else 0

            candidates.append({
                "Symbol": symbol,
                "Spot": round(spot, 2),
                "Strike": round(strike, 2),
                "Strike % OTM": otm_pct,
                "Expiry": exp_str,
                "DTE": dte,
                "Mid Price": round(mid, 2),
                "Implied Vol": float(best.get("impliedVolatility", np.nan)),
                "Volume": int(best.get("volume", 0) or 0),
                "Open Interest": int(best.get("openInterest", 0) or 0),
                "Contracts": contracts,
                "Premium Income": round(premium_total, 2),
                "If-Called Total Profit": round(if_called_total, 2),
                "Annualized Premium Yield": annualized_premium_yield,
                "Breakeven": round(spot - mid, 2),
            })
        return candidates
    except Exception:
        return []


def covered_call_screener(
    holdings: pd.DataFrame,
    *,
    target_dte_min: int = 25,
    target_dte_max: int = 50,
    otm_pct_target: float = 0.05,
    max_workers: int = 6,
) -> pd.DataFrame:
    """Run the covered-call screener across all stock holdings with ≥100 shares.

    `holdings` must have columns: Symbol, Shares, Asset Type.
    Only Stocks (not ETFs/MFs) with ≥100 shares are screened.
    """
    if holdings.empty or yf is None:
        return pd.DataFrame()
    eligible = holdings[
        (holdings["Asset Type"] == "Stocks")
        & (holdings["Shares"].fillna(0) >= 100)
    ][["Symbol", "Shares"]].dropna()
    if eligible.empty:
        return pd.DataFrame()

    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_covered_call_one, row["Symbol"], row["Shares"],
                      target_dte_min, target_dte_max, otm_pct_target)
            for _, row in eligible.iterrows()
        ]
        for f in as_completed(futures):
            all_rows.extend(f.result())

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    return df.sort_values("Annualized Premium Yield", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# IV-rank proxy (vs 252-day Black-Scholes implied vol estimate)
# ---------------------------------------------------------------------------
def realized_volatility(symbol: str, lookback_days: int = 30) -> float:
    """Compute realized (historical) volatility over `lookback_days`."""
    if yf is None:
        return np.nan
    try:
        hist = yf.Ticker(symbol).history(period="3mo", interval="1d", auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist:
            return np.nan
        rets = hist["Close"].pct_change().dropna().tail(lookback_days)
        if rets.empty:
            return np.nan
        return float(rets.std() * np.sqrt(252))
    except Exception:
        return np.nan


def iv_vs_realized(symbol: str) -> dict:
    """Compare ATM IV to recent realized volatility — proxy for 'is option pricey?'."""
    chain = get_options_chain(symbol)
    if not chain or chain["calls"].empty:
        return {}
    spot = chain["spot"]
    if np.isnan(spot):
        return {}
    calls = chain["calls"]
    # Find ATM call
    calls = calls.copy()
    calls["distance"] = (calls["strike"] - spot).abs()
    atm = calls.nsmallest(1, "distance").iloc[0]
    iv_atm = float(atm.get("impliedVolatility", np.nan))
    rv = realized_volatility(symbol, 30)
    return {
        "Symbol": symbol,
        "Spot": spot,
        "ATM IV": iv_atm,
        "30D Realized Vol": rv,
        "IV/RV Ratio": iv_atm / rv if rv and not np.isnan(rv) else np.nan,
        "Read": _iv_read(iv_atm, rv),
    }


def _iv_read(iv: float, rv: float) -> str:
    if np.isnan(iv) or np.isnan(rv) or rv <= 0:
        return "No data"
    ratio = iv / rv
    if ratio >= 1.5:
        return "Options expensive (IV >> RV) — favors selling premium"
    if ratio >= 1.1:
        return "Options modestly priced above realized — neutral/sell premium"
    if ratio >= 0.9:
        return "Options fairly priced relative to realized"
    return "Options cheap (IV < RV) — favors buying premium"


def iv_overview(symbols: Iterable[str], *, max_workers: int = 4) -> pd.DataFrame:
    """IV vs realized for many symbols in parallel."""
    syms = [s for s in symbols if str(s).strip()]
    if not syms or yf is None:
        return pd.DataFrame()
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(iv_vs_realized, s) for s in syms]
        for f in as_completed(futures):
            r = f.result()
            if r:
                rows.append(r)
    return pd.DataFrame(rows) if rows else pd.DataFrame()
