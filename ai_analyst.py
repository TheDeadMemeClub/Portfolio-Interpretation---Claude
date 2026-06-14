"""In-app AI analyst layer.

Wraps the Anthropic API so the dashboard can write the *narrative*: a plain-
English portfolio health check, factor-regression readouts, per-name bull/bear
theses, rebalance rationales, and news digests. The app does the data wrangling;
the model only turns structured numbers into prose.

Degrades gracefully: if the `anthropic` package or an API key is missing, every
function returns a helpful setup message instead of raising. Provide the key via
Streamlit secrets (`.streamlit/secrets.toml` -> ANTHROPIC_API_KEY = "sk-ant-...")
or the ANTHROPIC_API_KEY environment variable. The key never leaves your runtime.

Nothing here is financial advice — the system prompt frames every response as
educational analysis of the user's own holdings, with explicit hedging.
"""

from __future__ import annotations

import os

import pandas as pd

# Friendly label -> current API model string.
MODELS = {
    "Fast · Haiku 4.5": "claude-haiku-4-5",
    "Balanced · Sonnet 4.6": "claude-sonnet-4-6",
    "Deep · Opus 4.8": "claude-opus-4-8",
}
DEFAULT_MODEL = "claude-sonnet-4-6"

_SYSTEM = (
    "You are a sharp, concise buy-side analyst embedded in a retail investor's "
    "own portfolio dashboard. The user is quantitatively literate (DCF, factor "
    "models, CAPM, tax-lot mechanics) and prefers direct, no-fluff analysis. "
    "Write tight, specific commentary grounded only in the numbers you are given "
    "— never invent figures. Surface risks and tensions, not just positives. "
    "This is educational analysis of the user's own holdings, not personalized "
    "financial advice; you are not a financial advisor. Keep it skimmable: short "
    "paragraphs and the occasional bold lead-in, no padding."
)


# ---------------------------------------------------------------------------
# Client / availability
# ---------------------------------------------------------------------------
def _resolve_key() -> str | None:
    key = None
    try:
        import streamlit as st  # local import so the module is import-safe anywhere
        try:
            key = st.secrets.get("ANTHROPIC_API_KEY")  # type: ignore[attr-defined]
        except Exception:
            key = None
    except Exception:
        pass
    return key or os.environ.get("ANTHROPIC_API_KEY")


def availability() -> tuple[bool, str]:
    """Return (ready, message)."""
    try:
        import anthropic  # noqa: F401
    except Exception:
        return False, (
            "The `anthropic` package isn't installed. Add `anthropic` to "
            "requirements.txt and redeploy."
        )
    if not _resolve_key():
        return False, (
            "No API key found. Add `ANTHROPIC_API_KEY` to your Streamlit secrets "
            "(Settings → Secrets) or environment to enable AI commentary."
        )
    return True, "ready"


def _get_client():
    import anthropic
    return anthropic.Anthropic(api_key=_resolve_key())


def _call(prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = 1100) -> str:
    ready, msg = availability()
    if not ready:
        return f"🔌 **AI analyst not configured.** {msg}"
    try:
        client = _get_client()
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        ).strip() or "_(empty response)_"
    except Exception as exc:  # surface the error in the UI rather than crashing
        return f"⚠️ AI request failed: `{exc}`"


# ---------------------------------------------------------------------------
# Compact context builders (keep token cost low; feed numbers, not raw frames)
# ---------------------------------------------------------------------------
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
# Public analyst functions
# ---------------------------------------------------------------------------
def portfolio_commentary(
    summary: pd.DataFrame,
    allocation: pd.DataFrame,
    *,
    broker_total: float,
    ugl_total: float,
    hhi: float | None,
    grade: str | None = None,
    regime: str | None = None,
    model: str = DEFAULT_MODEL,
) -> str:
    alloc_lines = "\n".join(
        f"  - {r['Asset Class']}: {r['Weight']:.1%}" for _, r in allocation.iterrows()
    )
    prompt = (
        "Write a portfolio health check (≈180 words). Lead with the single most "
        "important takeaway, then concentration, then what to watch.\n\n"
        f"Total value: ${broker_total:,.0f}\n"
        f"Unrealized P&L: ${ugl_total:,.0f} ({_fmt_pct(ugl_total/broker_total) if broker_total else 'n/a'})\n"
        f"Concentration HHI: {hhi:.3f} (0.10 ≈ 10 equal names)\n"
        f"Portfolio grade: {grade or 'n/a'}\n"
        f"Market regime: {regime or 'n/a'}\n\n"
        f"Allocation:\n{alloc_lines}\n\n"
        f"Top holdings:\n{_top_holdings_block(summary)}\n"
    )
    return _call(prompt, model=model)


def explain_factors(
    factor_table: pd.DataFrame,
    *,
    alpha: float | None,
    r2: float | None,
    n_obs: int | None,
    model: str = DEFAULT_MODEL,
) -> str:
    rows = "\n".join(
        f"  - {r.iloc[0]}: beta {r.iloc[1]:+.2f}"
        + (f", t={r.iloc[2]:.1f}" if len(r) > 2 and pd.notna(r.iloc[2]) else "")
        for _, r in factor_table.iterrows()
    )
    prompt = (
        "Explain this factor regression in plain English for someone who knows "
        "what betas and t-stats mean but wants the intuition (≈160 words). Call "
        "out which exposures are statistically trustworthy vs noise, and what the "
        "R² implies about how much is left unexplained (idiosyncratic).\n\n"
        f"Annualized alpha: {_fmt_pct(alpha) if alpha is not None else 'n/a'}\n"
        f"R²: {r2:.2f}\n" if r2 is not None else "R²: n/a\n"
    ) + (
        f"Observations: {n_obs}\n\nFactor betas:\n{rows}\n"
    )
    return _call(prompt, model=model)


def thesis(
    symbol: str,
    fundamentals: dict,
    valuation: dict | None = None,
    *,
    model: str = DEFAULT_MODEL,
) -> str:
    def g(d, *keys):
        for k in keys:
            if d and k in d and pd.notna(d[k]):
                return d[k]
        return "n/a"

    prompt = (
        f"Give a tight bull/bear/verdict on {symbol} (≈170 words) using only "
        "these figures. End with a one-line verdict and the single biggest swing "
        "factor.\n\n"
        f"Sector: {g(fundamentals, 'Sector')}\n"
        f"P/E: {g(fundamentals, 'P/E', 'PE', 'Trailing P/E')}\n"
        f"Profit margin: {g(fundamentals, 'Profit Margin')}\n"
        f"Revenue growth: {g(fundamentals, 'Revenue Growth')}\n"
        f"FCF yield: {g(fundamentals, 'FCF Yield', 'Owner Earnings Yield')}\n"
        f"Debt/Equity: {g(fundamentals, 'Debt/Equity', 'D/E')}\n"
    )
    if valuation:
        prompt += (
            f"DCF fair value: {g(valuation, 'DCF Fair Value', 'Fair Value')}\n"
            f"Composite verdict: {g(valuation, 'Verdict', 'Composite Verdict')}\n"
        )
    return _call(prompt, model=model)


def rebalance_rationale(trades: pd.DataFrame, stats: dict, *, model: str = DEFAULT_MODEL) -> str:
    if trades is None or trades.empty:
        return "_No trades proposed — the book is within its target bands._"
    blotter = "\n".join(
        f"  - {r['Action']} {r.get('Shares','')} {r['Symbol']} "
        f"(~${r['Est. $']:,.0f}, {r.get('Term','')} {r.get('Rationale','')})"
        for _, r in trades.head(25).iterrows()
    )
    prompt = (
        "Explain this rebalance plan to the investor in plain English (≈150 "
        "words): what it accomplishes, why the lot choices are tax-smart, and the "
        "one caveat to double-check before trading.\n\n"
        f"Net cash freed: ${stats.get('net_cash_freed',0):,.0f}\n"
        f"Realized gains: ${stats.get('realized_gains',0):,.0f} | "
        f"realized losses: ${stats.get('realized_losses',0):,.0f}\n"
        f"Estimated tax: ${stats.get('est_tax',0):,.0f} | "
        f"tax saved vs gains-only: ${stats.get('tax_saved_vs_gains_only',0):,.0f}\n\n"
        f"Trades:\n{blotter}\n"
    )
    return _call(prompt, model=model)


def news_digest(symbol: str, headlines: list[dict], *, model: str = DEFAULT_MODEL) -> str:
    if not headlines:
        return "_No recent headlines to summarize._"
    items = "\n".join(
        f"  - {h.get('title', h.get('headline',''))}" for h in headlines[:10]
    )
    prompt = (
        f"Summarize the news flow on {symbol} (≈120 words): the net narrative, "
        "any catalyst or risk, and an overall lean (bullish / neutral / bearish) "
        "with a one-line why. Headlines only — don't fabricate detail beyond them.\n\n"
        f"{items}\n"
    )
    return _call(prompt, model=model)
