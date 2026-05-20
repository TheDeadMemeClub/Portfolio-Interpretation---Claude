"""Advanced valuation models.

For each stock, computes multiple valuation estimates and combines them into
a composite fair-value estimate + over/undervalued signal.

Methods implemented:
  * Discounted Cash Flow (DCF) - based on TTM FCF + growth assumption
  * Reverse-DCF implied growth - "what growth would justify today's price?"
  * Graham Number - sqrt(22.5 * EPS * BVPS), classic Ben Graham formula
  * PEG ratio - PE / growth, with classification
  * Multiple expansion / contraction - current PE vs 5Y avg PE
  * Owner Earnings yield - Buffett's preferred FCF metric
  * Composite fair-value verdict
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None  # type: ignore


# ---------------------------------------------------------------------------
# DCF model
# ---------------------------------------------------------------------------
def dcf_fair_value(
    fcf_ttm: float,
    shares_outstanding: float,
    growth_yr_1_5: float = 0.10,
    growth_yr_6_10: float = 0.04,
    terminal_growth: float = 0.025,
    discount_rate: float = 0.09,
    cash: float = 0.0,
    debt: float = 0.0,
) -> dict:
    """Two-stage DCF returning per-share fair value.

    Stage 1: years 1-5 grow at `growth_yr_1_5`
    Stage 2: years 6-10 grow at `growth_yr_6_10`
    Terminal: Gordon growth model

    Returns dict with fair_value_per_share, present_value_total, assumptions.
    """
    if fcf_ttm <= 0 or shares_outstanding <= 0 or discount_rate <= terminal_growth:
        return {"fair_value_per_share": np.nan, "error": "Invalid inputs"}

    pv_total = 0.0
    fcf = fcf_ttm

    # Years 1-5
    for year in range(1, 6):
        fcf *= (1 + growth_yr_1_5)
        pv_total += fcf / ((1 + discount_rate) ** year)

    # Years 6-10
    for year in range(6, 11):
        fcf *= (1 + growth_yr_6_10)
        pv_total += fcf / ((1 + discount_rate) ** year)

    # Terminal value at end of year 10
    terminal_fcf = fcf * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / ((1 + discount_rate) ** 10)
    pv_total += pv_terminal

    equity_value = pv_total + cash - debt
    per_share = equity_value / shares_outstanding

    return {
        "fair_value_per_share": float(per_share),
        "present_value_total": float(pv_total),
        "terminal_value_pv": float(pv_terminal),
        "terminal_pct_of_value": float(pv_terminal / pv_total) if pv_total else np.nan,
        "growth_yr_1_5": growth_yr_1_5,
        "growth_yr_6_10": growth_yr_6_10,
        "terminal_growth": terminal_growth,
        "discount_rate": discount_rate,
    }


def reverse_dcf_implied_growth(
    current_price: float,
    fcf_ttm: float,
    shares_outstanding: float,
    discount_rate: float = 0.09,
    terminal_growth: float = 0.025,
    terminal_pe: float = 15.0,
) -> float:
    """Iteratively solve for the FCF growth rate the market is pricing in.

    Returns the implied 10-year growth rate (decimal).
    """
    if current_price <= 0 or fcf_ttm <= 0 or shares_outstanding <= 0:
        return np.nan
    target_value = current_price * shares_outstanding

    # Binary search on growth rate
    lo, hi = -0.20, 0.50
    for _ in range(60):
        mid = (lo + hi) / 2
        pv_total = 0.0
        fcf = fcf_ttm
        for year in range(1, 11):
            fcf *= (1 + mid)
            pv_total += fcf / ((1 + discount_rate) ** year)
        terminal_fcf = fcf * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        pv_total += terminal_value / ((1 + discount_rate) ** 10)

        if pv_total > target_value:
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) < 0.0001:
            break
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Graham Number + classic value metrics
# ---------------------------------------------------------------------------
def graham_number(eps: float, bvps: float) -> float:
    """Ben Graham's intrinsic value formula: sqrt(22.5 * EPS * BVPS)."""
    if eps is None or bvps is None or eps <= 0 or bvps <= 0:
        return np.nan
    return math.sqrt(22.5 * eps * bvps)


def peg_ratio(pe: float, growth_pct: float) -> float:
    """PEG = P/E divided by growth rate (in percent, not decimal)."""
    if pe is None or growth_pct is None or pe <= 0 or growth_pct <= 0:
        return np.nan
    return pe / growth_pct


def peg_classification(peg: float) -> str:
    if pd.isna(peg) or peg <= 0:
        return "—"
    if peg < 1.0:
        return "Undervalued (PEG < 1)"
    if peg < 1.5:
        return "Fairly valued"
    if peg < 2.0:
        return "Slightly expensive"
    return "Expensive (PEG > 2)"


def owner_earnings_yield(fcf_ttm: float, market_cap: float) -> float:
    """FCF / Market Cap (Buffett's preferred yield metric)."""
    if not fcf_ttm or not market_cap or market_cap <= 0:
        return np.nan
    return fcf_ttm / market_cap


# ---------------------------------------------------------------------------
# Per-symbol fundamental fetch with valuation extras
# ---------------------------------------------------------------------------
def _fetch_valuation_one(symbol: str, *, dcf_growth: float, dcf_discount: float) -> dict:
    """Pull rich fundamentals + run all valuation models for one ticker."""
    out: dict = {"Symbol": symbol}
    if yf is None:
        out["Valuation Status"] = "yfinance not installed"
        return out
    try:
        t = yf.Ticker(symbol)
        try:
            info = t.get_info() or {}
        except Exception:
            info = {}
        try:
            fast = dict(t.fast_info or {})
        except Exception:
            fast = {}

        # Core inputs
        price = info.get("currentPrice") or info.get("regularMarketPrice") or fast.get("last_price")
        shares = info.get("sharesOutstanding") or fast.get("shares")
        eps = info.get("trailingEps")
        bvps = info.get("bookValue")
        pe = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        ps = info.get("priceToSalesTrailing12Months")
        pb = info.get("priceToBook")
        peg_yh = info.get("pegRatio") or info.get("trailingPegRatio")
        ebitda = info.get("ebitda")
        ev = info.get("enterpriseValue")
        ev_ebitda = info.get("enterpriseToEbitda")
        market_cap = info.get("marketCap") or fast.get("market_cap")
        fcf = info.get("freeCashflow")  # TTM
        op_cf = info.get("operatingCashflow")
        total_cash = info.get("totalCash") or 0
        total_debt = info.get("totalDebt") or 0
        rev_growth = info.get("revenueGrowth")
        earnings_growth = info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth")
        analyst_target = info.get("targetMeanPrice")
        analyst_high = info.get("targetHighPrice")
        analyst_low = info.get("targetLowPrice")
        recommendation = info.get("recommendationKey")
        roe = info.get("returnOnEquity")
        roic = info.get("returnOnAssets")
        gross_margin = info.get("grossMargins")
        op_margin = info.get("operatingMargins")
        debt_to_equity = info.get("debtToEquity")
        dividend_yield = info.get("dividendYield")
        payout_ratio = info.get("payoutRatio")
        beta = info.get("beta")
        sector = info.get("sector")
        industry = info.get("industry")

        out.update({
            "Sector": sector or "",
            "Industry": industry or "",
            "Price": price,
            "Market Cap": market_cap,
            "EPS (TTM)": eps,
            "BVPS": bvps,
            "Trailing P/E": pe,
            "Forward P/E": forward_pe,
            "Price/Sales": ps,
            "Price/Book": pb,
            "EV/EBITDA": ev_ebitda,
            "PEG (Yahoo)": peg_yh,
            "FCF (TTM)": fcf,
            "Op CF (TTM)": op_cf,
            "Revenue Growth": rev_growth,
            "Earnings Growth": earnings_growth,
            "ROE": roe,
            "ROIC (proxy)": roic,
            "Gross Margin": gross_margin,
            "Operating Margin": op_margin,
            "Debt/Equity": debt_to_equity,
            "Dividend Yield": dividend_yield,
            "Payout Ratio": payout_ratio,
            "Beta": beta,
            "Analyst Target": analyst_target,
            "Analyst High": analyst_high,
            "Analyst Low": analyst_low,
            "Recommendation": recommendation or "",
        })

        # --- Run all valuation models if we have what we need ---
        # 1. DCF
        if fcf and shares and price and fcf > 0 and shares > 0:
            dcf_result = dcf_fair_value(
                fcf, shares,
                growth_yr_1_5=dcf_growth, growth_yr_6_10=dcf_growth * 0.5,
                terminal_growth=0.025, discount_rate=dcf_discount,
                cash=total_cash, debt=total_debt,
            )
            fv = dcf_result.get("fair_value_per_share", np.nan)
            out["DCF Fair Value"] = fv
            if pd.notna(fv) and fv > 0:
                out["DCF Upside %"] = fv / price - 1
            # Reverse-DCF
            implied = reverse_dcf_implied_growth(price, fcf, shares,
                                                discount_rate=dcf_discount)
            out["Implied Growth (Reverse-DCF)"] = implied

        # 2. Graham Number
        if eps and bvps and eps > 0 and bvps > 0:
            gn = graham_number(eps, bvps)
            out["Graham Number"] = gn
            if price and price > 0:
                out["Graham Upside %"] = gn / price - 1

        # 3. PEG (own calc) — use earnings growth if available, else revenue growth
        gr = earnings_growth or rev_growth
        if pe and gr and gr > 0:
            peg = peg_ratio(pe, gr * 100)
            out["PEG (calc)"] = peg
            out["PEG Verdict"] = peg_classification(peg)

        # 4. Owner Earnings Yield
        if fcf and market_cap:
            out["Owner Earnings Yield"] = owner_earnings_yield(fcf, market_cap)

        # 5. Analyst implied upside
        if analyst_target and price:
            out["Analyst Upside %"] = analyst_target / price - 1

        # --- Composite verdict ---
        out["Valuation Verdict"] = _composite_verdict(out)

    except Exception as exc:
        out["Valuation Status"] = str(exc)[:120]
    return out


def _composite_verdict(row: dict) -> str:
    """Weighted combination of all valuation signals into one human verdict."""
    signals: list[float] = []  # Each in [-1, +1] where +1 = strongly undervalued

    # DCF upside
    dcf = row.get("DCF Upside %")
    if pd.notna(dcf):
        signals.append(np.clip(dcf, -1, 1))

    # Graham upside (weight half)
    gn = row.get("Graham Upside %")
    if pd.notna(gn):
        signals.append(np.clip(gn * 0.5, -1, 1))

    # PEG signal
    peg = row.get("PEG (calc)") or row.get("PEG (Yahoo)")
    if pd.notna(peg) and peg > 0:
        # PEG of 1 = neutral, 0.5 = strong buy, 2+ = expensive
        signals.append(np.clip((1.5 - peg) / 1.5, -1, 1))

    # Analyst target
    at = row.get("Analyst Upside %")
    if pd.notna(at):
        signals.append(np.clip(at, -1, 1))

    if not signals:
        return "—"
    score = float(np.mean(signals))
    if score > 0.30:
        return "Strongly undervalued"
    if score > 0.10:
        return "Undervalued"
    if score > -0.10:
        return "Fairly valued"
    if score > -0.30:
        return "Overvalued"
    return "Strongly overvalued"


def fetch_advanced_valuations(
    symbols: Iterable[str],
    *,
    dcf_growth: float = 0.10,
    dcf_discount: float = 0.09,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Pull advanced valuation data for many tickers in parallel."""
    syms = list(dict.fromkeys(str(s).upper().strip() for s in symbols if str(s).strip()))
    if not syms or yf is None:
        return pd.DataFrame({"Symbol": syms, "Valuation Status": "yfinance not available"})

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_fetch_valuation_one, s, dcf_growth=dcf_growth, dcf_discount=dcf_discount): s
            for s in syms
        }
        for f in as_completed(futures):
            try:
                rows.append(f.result())
            except Exception:
                pass
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Multiple expansion / contraction analysis
# ---------------------------------------------------------------------------
def historical_pe_band(symbol: str, years: int = 5) -> dict:
    """Estimate the 5-year P/E range for a stock using historical earnings.

    Useful for spotting multiple expansion / contraction.
    Returns dict with current_pe, avg_pe, min_pe, max_pe, current_vs_avg.
    """
    if yf is None:
        return {}
    try:
        t = yf.Ticker(symbol)
        info = t.get_info() or {}
        current_pe = info.get("trailingPE")
        try:
            # earnings history (annual)
            income = t.income_stmt
            if income is None or income.empty:
                return {"current_pe": current_pe}
        except Exception:
            return {"current_pe": current_pe}

        if "Diluted EPS" not in income.index and "Basic EPS" not in income.index:
            return {"current_pe": current_pe}
        eps_row = income.loc["Diluted EPS"] if "Diluted EPS" in income.index else income.loc["Basic EPS"]
        eps_series = eps_row.dropna()
        if eps_series.empty:
            return {"current_pe": current_pe}

        # For each historical year, find avg price during that year, divide by EPS
        hist_pes: list[float] = []
        for date_col, eps_val in eps_series.items():
            try:
                yr = pd.Timestamp(date_col).year
                year_prices = t.history(start=f"{yr}-01-01", end=f"{yr}-12-31",
                                       interval="1d", auto_adjust=False)
                if year_prices is None or year_prices.empty or eps_val <= 0:
                    continue
                avg_price = float(year_prices["Close"].mean())
                hist_pes.append(avg_price / float(eps_val))
            except Exception:
                continue

        if not hist_pes:
            return {"current_pe": current_pe}

        avg_pe = float(np.mean(hist_pes))
        min_pe = float(np.min(hist_pes))
        max_pe = float(np.max(hist_pes))
        return {
            "current_pe": current_pe,
            "avg_pe": avg_pe,
            "min_pe": min_pe,
            "max_pe": max_pe,
            "current_vs_avg": (current_pe / avg_pe - 1) if current_pe and avg_pe else np.nan,
            "n_years": len(hist_pes),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Quality screen
# ---------------------------------------------------------------------------
def quality_screen(valuation_df: pd.DataFrame) -> pd.DataFrame:
    """Score each stock on quality: ROE > 15%, gross margin > 40%, low debt, growing.

    Classic Buffett-style quality screen. Returns the df with a Quality Score
    (0-100) and Quality Tier column appended.
    """
    if valuation_df.empty:
        return valuation_df
    df = valuation_df.copy()

    def _score(row: pd.Series) -> float:
        score = 0.0
        # ROE
        roe = row.get("ROE")
        if pd.notna(roe):
            if roe > 0.25:
                score += 25
            elif roe > 0.15:
                score += 18
            elif roe > 0.10:
                score += 10
            elif roe > 0:
                score += 5
        # Gross margin
        gm = row.get("Gross Margin")
        if pd.notna(gm):
            if gm > 0.50:
                score += 20
            elif gm > 0.40:
                score += 15
            elif gm > 0.25:
                score += 8
            elif gm > 0:
                score += 3
        # Operating margin
        om = row.get("Operating Margin")
        if pd.notna(om):
            if om > 0.25:
                score += 15
            elif om > 0.15:
                score += 10
            elif om > 0.05:
                score += 5
        # Revenue growth
        rg = row.get("Revenue Growth")
        if pd.notna(rg):
            if rg > 0.20:
                score += 15
            elif rg > 0.10:
                score += 10
            elif rg > 0.03:
                score += 5
        # Debt-to-equity (lower is better; 0-50 is great, >150 is bad)
        de = row.get("Debt/Equity")
        if pd.notna(de):
            if de < 30:
                score += 15
            elif de < 75:
                score += 10
            elif de < 150:
                score += 5
        # Owner earnings yield (>5% = good)
        oey = row.get("Owner Earnings Yield")
        if pd.notna(oey):
            if oey > 0.07:
                score += 10
            elif oey > 0.05:
                score += 7
            elif oey > 0.03:
                score += 4

        return min(score, 100)

    df["Quality Score"] = df.apply(_score, axis=1).round(1)
    df["Quality Tier"] = pd.cut(
        df["Quality Score"],
        bins=[-0.01, 30, 55, 75, 100.01],
        labels=["Low", "Average", "Good", "Excellent"],
    ).astype(str)
    return df
