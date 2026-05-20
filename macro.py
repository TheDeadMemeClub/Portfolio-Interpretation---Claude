"""Macro economic indicators + insider trading + institutional ownership.

Pulls big-picture market context that helps frame whether your portfolio
should be aggressive or defensive right now.

Indicators tracked:
  * VIX (fear index)
  * 10Y Treasury yield
  * 2Y Treasury yield + 2s10s spread (recession signal)
  * Gold (haven asset)
  * DXY (dollar index)
  * High-yield credit spread proxy (HYG vs LQD)
  * 50-day vs 200-day SPY for trend regime
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


MACRO_TICKERS: dict[str, str] = {
    "VIX (Fear)":      "^VIX",
    "10Y Treasury":    "^TNX",
    "2Y Treasury":     "^IRX",  # 13-week T-bill as 2Y proxy via yfinance
    "30Y Treasury":    "^TYX",
    "Gold":            "GLD",
    "Dollar Index":    "DX-Y.NYB",
    "Oil (WTI)":       "CL=F",
    "S&P 500":         "SPY",
    "Nasdaq 100":      "QQQ",
    "Russell 2000":    "IWM",
    "High Yield Bond": "HYG",
    "Inv Grade Bond":  "LQD",
}


def _fetch_macro_one(name_ticker: tuple[str, str]) -> dict | None:
    name, ticker = name_ticker
    if yf is None:
        return None
    try:
        hist = yf.Ticker(ticker).history(period="6mo", interval="1d", auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist:
            return None
        close = hist["Close"].dropna()
        if len(close) < 5:
            return None
        current = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) >= 2 else current
        wk_ago = float(close.iloc[-5]) if len(close) >= 5 else current
        mo_ago = float(close.iloc[-21]) if len(close) >= 21 else current
        ytd_start = close[close.index >= f"{close.index[-1].year}-01-01"]
        ytd_anchor = float(ytd_start.iloc[0]) if not ytd_start.empty else current
        return {
            "Indicator": name,
            "Ticker": ticker,
            "Current": current,
            "1D Change": (current / prev - 1) if prev else 0,
            "1W Change": (current / wk_ago - 1) if wk_ago else 0,
            "1M Change": (current / mo_ago - 1) if mo_ago else 0,
            "YTD Change": (current / ytd_anchor - 1) if ytd_anchor else 0,
            "6M High": float(close.max()),
            "6M Low": float(close.min()),
        }
    except Exception:
        return None


def fetch_macro_dashboard(*, max_workers: int = 6) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_fetch_macro_one, item) for item in MACRO_TICKERS.items()]
        for f in as_completed(futures):
            r = f.result()
            if r:
                rows.append(r)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Restore the intended order
    order = list(MACRO_TICKERS.keys())
    df["__order"] = df["Indicator"].map({n: i for i, n in enumerate(order)})
    return df.sort_values("__order").drop(columns=["__order"]).reset_index(drop=True)


def market_regime_signal() -> dict:
    """Determine current market regime based on SPY trend + VIX + yield curve.

    Returns a dict with regime label and reasoning.
    """
    if yf is None:
        return {}
    try:
        spy = yf.Ticker("SPY").history(period="1y", interval="1d", auto_adjust=False)
        vix = yf.Ticker("^VIX").history(period="1mo", interval="1d", auto_adjust=False)
        tnx = yf.Ticker("^TNX").history(period="1mo", interval="1d", auto_adjust=False)
        irx = yf.Ticker("^IRX").history(period="1mo", interval="1d", auto_adjust=False)

        if spy.empty:
            return {}
        close = spy["Close"]
        sma50 = float(close.rolling(50).mean().iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1])
        current = float(close.iloc[-1])

        vix_now = float(vix["Close"].iloc[-1]) if not vix.empty else np.nan
        tnx_now = float(tnx["Close"].iloc[-1]) if not tnx.empty else np.nan
        irx_now = float(irx["Close"].iloc[-1]) if not irx.empty else np.nan

        # Score it
        risk_on_signals = 0
        risk_off_signals = 0
        notes = []

        if current > sma50 > sma200:
            risk_on_signals += 2
            notes.append("SPY in confirmed uptrend (above 50 > 200 SMA)")
        elif current > sma200:
            risk_on_signals += 1
            notes.append("SPY above 200-day SMA")
        elif current < sma200:
            risk_off_signals += 2
            notes.append("SPY below 200-day SMA (cyclical bear signal)")

        if not np.isnan(vix_now):
            if vix_now < 15:
                risk_on_signals += 1
                notes.append(f"VIX low at {vix_now:.1f} (complacency)")
            elif vix_now > 25:
                risk_off_signals += 1
                notes.append(f"VIX elevated at {vix_now:.1f} (stress)")

        if not np.isnan(tnx_now) and not np.isnan(irx_now):
            spread = tnx_now - irx_now
            if spread < 0:
                risk_off_signals += 2
                notes.append(f"Yield curve inverted ({spread:.2f}pp) - recession signal")
            elif spread < 0.5:
                notes.append(f"Yield curve flat ({spread:.2f}pp) - caution")
            else:
                notes.append(f"Yield curve normal ({spread:.2f}pp)")

        # Determine regime
        if risk_on_signals >= 3 and risk_off_signals == 0:
            regime = "Risk-On / Bullish"
            color = "bullish"
        elif risk_off_signals >= 3:
            regime = "Risk-Off / Bearish"
            color = "bearish"
        elif risk_off_signals > risk_on_signals:
            regime = "Defensive Tilt"
            color = "warn"
        else:
            regime = "Neutral / Mixed"
            color = "neutral"

        return {
            "regime": regime,
            "color": color,
            "score_on": risk_on_signals,
            "score_off": risk_off_signals,
            "notes": notes,
            "spy_current": current,
            "spy_sma50": sma50,
            "spy_sma200": sma200,
            "vix": vix_now,
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Insider activity
# ---------------------------------------------------------------------------
def _insider_one(symbol: str) -> dict | None:
    if yf is None:
        return None
    try:
        t = yf.Ticker(symbol)
        try:
            ins = t.insider_transactions
        except Exception:
            ins = None
        if ins is None or ins.empty:
            return None

        # Sum recent buys vs sells (last ~6 months)
        ins = ins.copy()
        if "Start Date" in ins.columns:
            ins["Start Date"] = pd.to_datetime(ins["Start Date"], errors="coerce")
            cutoff = pd.Timestamp.today() - pd.Timedelta(days=180)
            recent = ins[ins["Start Date"] >= cutoff]
        else:
            recent = ins

        if recent.empty:
            return None

        # The columns differ — try common ones
        text_col = next((c for c in ["Text", "Action", "Transaction"] if c in recent.columns), None)
        value_col = next((c for c in ["Value", "Transaction Value"] if c in recent.columns), None)
        shares_col = next((c for c in ["Shares", "Trade Shares"] if c in recent.columns), None)

        n_buys = n_sells = 0
        buy_value = sell_value = 0.0
        if text_col:
            for _, row in recent.iterrows():
                txt = str(row.get(text_col, "")).upper()
                val = float(row.get(value_col, 0) or 0) if value_col else 0
                if "BUY" in txt or "PURCHAS" in txt:
                    n_buys += 1
                    buy_value += val
                elif "SALE" in txt or "SELL" in txt or "DISPOS" in txt:
                    n_sells += 1
                    sell_value += val

        net = buy_value - sell_value
        return {
            "Symbol": symbol,
            "Insider Buys (180d)": n_buys,
            "Insider Sells (180d)": n_sells,
            "Buy Value": buy_value,
            "Sell Value": sell_value,
            "Net Insider Activity $": net,
            "Bias": "Buying" if net > 0 else ("Selling" if net < 0 else "Neutral"),
        }
    except Exception:
        return None


def fetch_insider_activity(symbols: Iterable[str], *, max_workers: int = 4) -> pd.DataFrame:
    syms = [s for s in symbols if str(s).strip()]
    if not syms or yf is None:
        return pd.DataFrame()
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_insider_one, s) for s in syms]
        for f in as_completed(futures):
            r = f.result()
            if r:
                rows.append(r)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Short interest snapshot
# ---------------------------------------------------------------------------
def short_interest_snapshot(symbols: Iterable[str], *, max_workers: int = 4) -> pd.DataFrame:
    """Pull short interest data via yfinance .info for many tickers."""
    syms = [s for s in symbols if str(s).strip()]
    if not syms or yf is None:
        return pd.DataFrame()

    def _one(sym: str) -> dict | None:
        try:
            info = yf.Ticker(sym).get_info() or {}
            short_pct = info.get("shortPercentOfFloat")
            short_ratio = info.get("shortRatio")
            if short_pct is None and short_ratio is None:
                return None
            return {
                "Symbol": sym,
                "Short % of Float": short_pct,
                "Days to Cover": short_ratio,
                "Shares Short": info.get("sharesShort"),
                "Short Prior Month": info.get("sharesShortPriorMonth"),
                "Squeeze Risk": _classify_squeeze(short_pct, short_ratio),
            }
        except Exception:
            return None

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_one, s) for s in syms]
        for f in as_completed(futures):
            r = f.result()
            if r:
                rows.append(r)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _classify_squeeze(short_pct: float | None, short_ratio: float | None) -> str:
    if short_pct is None:
        return "—"
    if short_pct > 0.20 and (short_ratio or 0) > 5:
        return "HIGH"
    if short_pct > 0.10:
        return "Elevated"
    if short_pct > 0.05:
        return "Moderate"
    return "Low"


# ---------------------------------------------------------------------------
# Analyst rating change tracker
# ---------------------------------------------------------------------------
def fetch_analyst_changes(symbols: Iterable[str], *, max_workers: int = 4) -> pd.DataFrame:
    """Recent analyst upgrades/downgrades for portfolio holdings."""
    syms = [s for s in symbols if str(s).strip()]
    if not syms or yf is None:
        return pd.DataFrame()

    def _one(sym: str) -> list[dict]:
        try:
            t = yf.Ticker(sym)
            up = t.upgrades_downgrades
            if up is None or up.empty:
                return []
            up = up.copy()
            up["Symbol"] = sym
            # Take most recent 5
            return up.head(5).reset_index().to_dict("records")
        except Exception:
            return []

    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_one, s) for s in syms]
        for f in as_completed(futures):
            all_rows.extend(f.result())
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    return df
