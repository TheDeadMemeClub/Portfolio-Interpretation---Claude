"""UI helpers — theme, reusable components, formatters.

NEW module. Centralizes:
  * Custom CSS / theme injection
  * Color palette
  * Formatter helpers (currency, percent, number)
  * Reusable Streamlit components (metric card, risk pill, status badge)
"""
from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
PALETTE = {
    "bg":             "#0E0E11",
    "surface":        "#16161A",
    "surface_2":      "#1F1F25",
    "border":         "#2A2A33",
    "text":           "#E6E6EA",
    "text_muted":     "#9A9AA8",
    "primary":        "#7F77DD",
    "primary_2":      "#534AB7",
    "bullish":        "#22C55E",
    "bearish":        "#EF4444",
    "neutral":        "#94A3B8",
    "warning":        "#F59E0B",
    "info":           "#38BDF8",
}


def inject_theme() -> None:
    """Inject custom CSS to give the dashboard a polished, consistent look."""
    st.markdown(
        f"""
        <style>
        :root {{
            --epic-primary: {PALETTE['primary']};
            --epic-bullish: {PALETTE['bullish']};
            --epic-bearish: {PALETTE['bearish']};
            --epic-border: {PALETTE['border']};
            --epic-muted: {PALETTE['text_muted']};
        }}
        .block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; }}

        /* Metric tweaks */
        div[data-testid="stMetricValue"] {{ font-size: 1.55rem; font-weight: 600; }}
        div[data-testid="stMetricDelta"] {{ font-size: 0.85rem; font-weight: 500; }}
        div[data-testid="stMetricLabel"] {{ color: {PALETTE['text_muted']}; font-size: 0.78rem; }}

        /* Card-ish containers for grouped content */
        .epic-card {{
            border: 1px solid {PALETTE['border']};
            border-radius: 14px;
            padding: 18px 20px;
            background: {PALETTE['surface']};
            margin-bottom: 1rem;
        }}

        /* Risk / status pills */
        .epic-pill {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.02em;
        }}
        .pill-bullish {{ background: rgba(34,197,94,0.18); color: {PALETTE['bullish']}; }}
        .pill-bearish {{ background: rgba(239,68,68,0.18); color: {PALETTE['bearish']}; }}
        .pill-neutral {{ background: rgba(148,163,184,0.18); color: {PALETTE['neutral']}; }}
        .pill-warn    {{ background: rgba(245,158,11,0.18); color: {PALETTE['warning']}; }}
        .pill-info    {{ background: rgba(56,189,248,0.18); color: {PALETTE['info']}; }}

        /* Tighter tabs */
        button[data-baseweb="tab"] {{ padding: 6px 14px; }}

        /* Dataframe header tweak (uses Streamlit default but cleaner) */
        thead tr th {{ background: {PALETTE['surface_2']} !important; }}

        /* Better mobile spacing */
        @media (max-width: 768px) {{
            .block-container {{ padding-left: 0.6rem; padding-right: 0.6rem; }}
            div[data-testid="stMetricValue"] {{ font-size: 1.2rem; }}
        }}

        /* Subtle headline */
        .epic-headline {{
            background: linear-gradient(135deg, {PALETTE['primary']} 0%, {PALETTE['primary_2']} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 700;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------
def fmt_currency(x, decimals: int = 0, prefix: str = "$") -> str:
    if x is None:
        return ""
    try:
        if x != x:  # NaN
            return ""
        return f"{prefix}{x:,.{decimals}f}"
    except (TypeError, ValueError):
        return str(x)


def fmt_percent(x, decimals: int = 2, signed: bool = False) -> str:
    if x is None:
        return ""
    try:
        if x != x:
            return ""
        fmt = f"{{:+.{decimals}%}}" if signed else f"{{:.{decimals}%}}"
        return fmt.format(x)
    except (TypeError, ValueError):
        return str(x)


def fmt_number(x, decimals: int = 2) -> str:
    if x is None:
        return ""
    try:
        if x != x:
            return ""
        return f"{x:,.{decimals}f}"
    except (TypeError, ValueError):
        return str(x)


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
def pill(text: str, kind: str = "neutral") -> str:
    """Return an HTML pill for inline status indicators.

    kind: bullish | bearish | neutral | warn | info
    """
    kind_class = f"pill-{kind}"
    return f'<span class="epic-pill {kind_class}">{text}</span>'


def trend_pill(value: str) -> str:
    if value == "Bullish":
        return pill("Bullish", "bullish")
    if value == "Bearish":
        return pill("Bearish", "bearish")
    if value == "Neutral":
        return pill("Neutral", "neutral")
    return pill(value or "No data", "neutral")


def trend_badge_color(value: str) -> str:
    """Inline CSS for Styler.map() coloring of trend cells."""
    if value == "Bullish":
        return f"background-color: rgba(34,197,94,0.30); color: white; font-weight: 700"
    if value == "Bearish":
        return f"background-color: rgba(239,68,68,0.30); color: white; font-weight: 700"
    if value == "Neutral":
        return f"background-color: rgba(148,163,184,0.18); color: white"
    return ""


def risk_tier_color(value: str) -> str:
    mapping = {
        "Low":      "background-color: rgba(34,197,94,0.20); color: white",
        "Moderate": "background-color: rgba(56,189,248,0.20); color: white",
        "Elevated": "background-color: rgba(245,158,11,0.25); color: white",
        "High":     "background-color: rgba(239,68,68,0.30); color: white; font-weight: 700",
    }
    return mapping.get(str(value), "")


def style_cells(df, *, trend_cols=None, risk_cols=None):
    """Helper that handles pandas Styler.map vs applymap compatibility."""
    styler = df.style
    apply_fn = styler.map if hasattr(styler, "map") else styler.applymap
    if trend_cols:
        styler = apply_fn(trend_badge_color, subset=[c for c in trend_cols if c in df.columns])
        apply_fn = styler.map if hasattr(styler, "map") else styler.applymap
    if risk_cols:
        styler = apply_fn(risk_tier_color, subset=[c for c in risk_cols if c in df.columns])
    return styler


def section_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(
        f"""
        <div style="margin-top:0.25rem;margin-bottom:0.4rem">
            <span style="font-weight:600;font-size:1.05rem;color:{PALETTE['text']}">{title}</span>
            {f'<div style="color:{PALETTE["text_muted"]};font-size:0.85rem;margin-top:2px">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(message: str, icon: str = "📊") -> None:
    st.markdown(
        f"""
        <div class="epic-card" style="text-align:center;padding:32px 16px">
            <div style="font-size:32px;margin-bottom:8px">{icon}</div>
            <div style="color:{PALETTE['text_muted']};font-size:0.92rem">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
