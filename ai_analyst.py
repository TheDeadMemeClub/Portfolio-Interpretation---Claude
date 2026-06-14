"""Builds copy-paste-ready analyst prompts from live portfolio data.

No API calls and no key required. The app does the tedious part — packing your
live holdings, factor output, valuations, or a rebalance blotter into a tight,
self-contained brief — and you paste the result straight into Claude (or any
assistant) to get the narrative for free using a subscription you already have.

Every function returns a finished prompt string: a role/instruction line plus the
data block, so it "just works" when pasted into a fresh chat.
"""

from __future__ import annotations

import pandas as pd

_ROLE = (
    "You are a sharp, concise buy-side analyst. I'm a quantitatively literate "
    "retail investor (DCF, factor models, tax-lot mechanics) — give me direct, "
    "no-fluff analysis grounded only in the numbers below. Surface risks and "
    "tensions, not just positives. This is educational analysis of my own "
    "holdings, not advice."
)


def _fmt_pct(x) -> str:
    try:
        return f"{float(x):+.1%}"
    except Exception:
        return "n/a"


def _top_holdings_block(summary: pd.DataFrame, n: int = 12) -> str:
    cols = [c for c in ["Symbol", "Asset Type", "Portfolio Weight",
                        "Unrealized P&L %", "Trend Rating", "Risk Tier"]
            if c in summary.columns]
    top = summary.sort_values("Market Value", ascending=False).head(n)[cols]
    lines = []
    for _, r in top.iterrows():
        parts = [str(r.get("Symbol", ""))]
        if "Portfolio Weight" in cols:
            parts.append(f"{r['Portfolio Weight']:.1%}")
        if "Unrealized P&L %" in cols and pd.notna(r.get("Unrealized P&L %")):
            parts.append(f"UGL {_fmt_pct(r['Unrealized P&L %'])}")
        if "Trend Rating" in cols and pd.notna(r.get("Trend Rating")):
            parts.append(str(r["Trend Rating"]))
        if "Risk Tier" in cols and pd.notna(r.get("Risk Tier")):
            parts.append(f"{r['Risk Tier']} risk")
        lines.append("  - " + " | ".join(parts))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def portfolio_briefing_prompt(
    summary: pd.DataFrame,
    allocation: pd.DataFrame,
    *,
    broker_total: float,
    ugl_total: float,
    hhi: float | None,
    grade: str | None = None,
    regime: str | None = None,
) -> str:
    alloc_lines = "\n".join(
        f"  - {r['Asset Class']}: {r['Weight']:.1%}" for _, r in allocation.iterrows()
    )
    return (
        f"{_ROLE}\n\n"
        "TASK: Write a portfolio health check (~180 words). Lead with the single "
        "most important takeaway, then concentration, then what to watch.\n\n"
        "PORTFOLIO\n"
        f"Total value: ${broker_total:,.0f}\n"
        f"Unrealized P&L: ${ugl_total:,.0f} "
        f"({_fmt_pct(ugl_total/broker_total) if broker_total else 'n/a'})\n"
        f"Concentration HHI: {hhi:.3f} (0.10 ≈ 10 equal names)\n"
        + (f"Portfolio grade: {grade}\n" if grade else "")
        + (f"Market regime: {regime}\n" if regime else "")
        + f"\nAllocation:\n{alloc_lines}\n\n"
        f"Top holdings:\n{_top_holdings_block(summary)}\n"
    )


def factor_prompt(
    factor_table: pd.DataFrame,
    *,
    alpha: float | None,
    r2: float | None,
    n_obs: int | None,
) -> str:
    rows = "\n".join(
        f"  - {r.iloc[0]}: beta {r.iloc[1]:+.2f}"
        + (f", t={r.iloc[2]:.1f}" if len(r) > 2 and pd.notna(r.iloc[2]) else "")
        for _, r in factor_table.iterrows()
    )
    return (
        f"{_ROLE}\n\n"
        "TASK: Explain this factor regression in plain English (~160 words). Call "
        "out which exposures are statistically trustworthy vs noise (using the "
        "t-stats), and what the R² implies about how much is left unexplained "
        "(idiosyncratic).\n\n"
        "REGRESSION\n"
        f"Annualized alpha: {_fmt_pct(alpha) if alpha is not None else 'n/a'}\n"
        f"R²: {r2:.2f}\n" if r2 is not None else "R²: n/a\n"
    ) + f"Observations: {n_obs}\n\nFactor betas:\n{rows}\n"


def thesis_prompt(symbol: str, fundamentals: dict) -> str:
    def g(*keys):
        for k in keys:
            if fundamentals and k in fundamentals and pd.notna(fundamentals[k]):
                return fundamentals[k]
        return "n/a"

    return (
        f"{_ROLE}\n\n"
        f"TASK: Give a tight bull/bear/verdict on {symbol} (~170 words) using only "
        "these figures. End with a one-line verdict and the single biggest swing "
        "factor.\n\n"
        f"{symbol}\n"
        f"Sector: {g('Sector')}\n"
        f"P/E: {g('P/E', 'PE', 'Trailing P/E')}\n"
        f"Profit margin: {g('Profit Margin')}\n"
        f"Revenue growth: {g('Revenue Growth')}\n"
        f"FCF yield: {g('FCF Yield', 'Owner Earnings Yield')}\n"
        f"Debt/Equity: {g('Debt/Equity', 'D/E')}\n"
        f"Portfolio weight: {g('Portfolio Weight')}\n"
        f"Unrealized P&L %: {g('Unrealized P&L %')}\n"
    )


def rebalance_prompt(trades: pd.DataFrame, stats: dict) -> str:
    if trades is None or trades.empty:
        return "No trades were proposed — the book is within its target bands."
    blotter = "\n".join(
        f"  - {r['Action']} {r.get('Shares','')} {r['Symbol']} "
        f"(~${r['Est. $']:,.0f}, {r.get('Term','')} {r.get('Rationale','')})"
        for _, r in trades.head(25).iterrows()
    )
    return (
        f"{_ROLE}\n\n"
        "TASK: Explain this rebalance plan in plain English (~150 words): what it "
        "accomplishes, why the lot choices are tax-smart, and the one caveat to "
        "double-check before trading.\n\n"
        "PLAN\n"
        f"Net cash freed: ${stats.get('net_cash_freed',0):,.0f}\n"
        f"Realized gains: ${stats.get('realized_gains',0):,.0f} | "
        f"realized losses: ${stats.get('realized_losses',0):,.0f}\n"
        f"Estimated tax: ${stats.get('est_tax',0):,.0f} | "
        f"tax saved vs gains-only: ${stats.get('tax_saved_vs_gains_only',0):,.0f}\n\n"
        f"Trades:\n{blotter}\n"
    )


def news_prompt(symbol: str, headlines: list[dict]) -> str:
    if not headlines:
        return f"No recent headlines available for {symbol}."
    items = "\n".join(
        f"  - {h.get('title', h.get('headline',''))}" for h in headlines[:10]
    )
    return (
        f"{_ROLE}\n\n"
        f"TASK: Summarize the news flow on {symbol} (~120 words): the net "
        "narrative, any catalyst or risk, and an overall lean (bullish / neutral "
        "/ bearish) with a one-line why. Use only the headlines below.\n\n"
        f"HEADLINES\n{items}\n"
    )
