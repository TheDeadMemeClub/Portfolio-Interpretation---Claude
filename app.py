"""Portfolio EPIC — main Streamlit app.

A private advisor-grade dashboard built around the same workflow as the original:
upload a broker positions export, get a full analysis. This version keeps every
feature from the original app and adds:

  * Smart Money / M-Block analyzer integrated as a tab + auto-runs on top
    holdings (the original `smart_money_feature.py` was never wired in).
  * Tax-loss harvesting tab — surfaces losing lots, segregates LT vs ST.
  * Benchmark comparison — portfolio vs SPY cumulative-return chart + alpha/beta.
  * Per-position risk scores (0-100 composite).
  * News headlines for holdings.
  * Sector concentration / HHI breakdown.
  * Parallel market-data fetching (3-5x faster on bigger portfolios).
  * Polished dark theme + PDF dashboard export.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Project modules
from parser import parse_broker_upload, tradingview_csv
from market_data import (
    fetch_fundamentals, fetch_returns, fetch_technical_snapshot,
    max_drawdown_from_returns, drawdown_series, fetch_news, fetch_earnings_dates,
    fetch_dividend_history, fetch_price_window, fetch_factor_returns,
)
from analytics import (
    compute_position_risk_scores, find_tlh_candidates, tlh_summary,
    benchmark_comparison, benchmark_stats, sector_concentration_hhi,
    correlation_clusters, auto_smart_money_table,
)
from advanced_analytics import (
    performance_attribution, monte_carlo_projection, probability_of_target,
    historical_stress_test, factor_exposure, efficient_frontier,
    portfolio_grade, dividend_growth_analysis, FACTOR_PROXIES,
)
from options_analysis import covered_call_screener, iv_overview
from backtest import backtest_portfolio
from valuation import fetch_advanced_valuations, quality_screen, historical_pe_band
from themes import (
    assign_themes, theme_exposure, theme_concentration_hhi,
    suggest_rebalance_actions, simulate_rebalance, RebalanceAction,
)
from macro import (
    fetch_macro_dashboard, market_regime_signal, fetch_insider_activity,
    short_interest_snapshot, fetch_analyst_changes,
)
from smart_money import render_smart_money_tab
from exports import df_to_csv_bytes, build_zip, build_pdf_report, pdf_available
from ui_helpers import (
    PALETTE, inject_theme, fmt_currency, fmt_percent,
    trend_pill, style_cells, section_header, empty_state,
)
from rebalance_engine import generate_plan
from signals import build_signals, signal_counts
import ai_analyst


# =============================================================================
# Page config + theme
# =============================================================================
st.set_page_config(
    page_title="Portfolio EPIC",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()


# =============================================================================
# Cached wrappers
# =============================================================================
@st.cache_data(ttl=900, show_spinner=False)
def cached_technical_snapshot(symbols_tuple, period, interval):
    return fetch_technical_snapshot(list(symbols_tuple), period=period, interval=interval)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_fundamentals(symbols_tuple):
    return fetch_fundamentals(list(symbols_tuple))


@st.cache_data(ttl=900, show_spinner=False)
def cached_returns(symbols_tuple, period):
    return fetch_returns(list(symbols_tuple), period=period)


@st.cache_data(ttl=900, show_spinner=False)
def cached_news(symbol: str):
    return fetch_news(symbol, max_items=6)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_earnings(symbols_tuple):
    return fetch_earnings_dates(list(symbols_tuple))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_factor_returns(period: str = "2y"):
    return fetch_factor_returns(FACTOR_PROXIES, period=period)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_dividend_growth(symbols_tuple):
    return dividend_growth_analysis(list(symbols_tuple), fetch_dividend_history)


@st.cache_data(ttl=900, show_spinner=False)
def cached_covered_calls(symbols_tuple, shares_tuple, dte_min, dte_max, otm_target):
    holdings = pd.DataFrame({"Symbol": list(symbols_tuple), "Shares": list(shares_tuple),
                             "Asset Type": ["Stocks"] * len(symbols_tuple)})
    return covered_call_screener(holdings, target_dte_min=dte_min,
                                target_dte_max=dte_max, otm_pct_target=otm_target)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_iv_overview(symbols_tuple):
    return iv_overview(list(symbols_tuple))


@st.cache_data(ttl=900, show_spinner=False)
def cached_backtest(symbols_tuple, weights_tuple, start, end, benchmark, starting_value, rebalance):
    w = pd.Series(list(weights_tuple), index=list(symbols_tuple))
    return backtest_portfolio(list(symbols_tuple), w, start=start, end=end,
                             benchmark=benchmark, starting_value=starting_value,
                             rebalance=rebalance)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_stress_test(symbols_tuple, weights_tuple, benchmark):
    w = pd.Series(list(weights_tuple), index=list(symbols_tuple))
    return historical_stress_test(list(symbols_tuple), w, fetch_price_window, benchmark=benchmark)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_advanced_valuations(symbols_tuple, growth, discount):
    df = fetch_advanced_valuations(list(symbols_tuple), dcf_growth=growth, dcf_discount=discount)
    return quality_screen(df) if not df.empty else df


@st.cache_data(ttl=900, show_spinner=False)
def cached_macro():
    return fetch_macro_dashboard()


@st.cache_data(ttl=900, show_spinner=False)
def cached_regime():
    return market_regime_signal()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_insider(symbols_tuple):
    return fetch_insider_activity(list(symbols_tuple))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_short_interest(symbols_tuple):
    return short_interest_snapshot(list(symbols_tuple))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_analyst_changes(symbols_tuple):
    return fetch_analyst_changes(list(symbols_tuple))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_pe_band(symbol):
    return historical_pe_band(symbol)


# =============================================================================
# Header
# =============================================================================
st.markdown(
    f'<h1 class="epic-headline" style="margin-bottom:0">📊 Portfolio EPIC</h1>',
    unsafe_allow_html=True,
)
st.caption(
    "Upload a broker positions export and turn it into a private advisor-grade dashboard: "
    "allocation, heat maps, trend ratings, valuation, income, risk, tax lots, "
    "smart-money setups, tax-loss harvesting, and exports."
)


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("Broker positions file", type=["xls", "xlsx", "csv"])
    st.caption(
        "Optimized for Wells Fargo Advisors position exports. Your broker file is "
        "only used at runtime — never bundled in the app."
    )

    st.header("Dashboard controls")
    color_by = st.radio(
        "Heat map color by",
        ["Today's Change %", "Unrealized P&L %", "Portfolio Weight", "Risk Score"],
        horizontal=False,
    )
    min_tile = st.slider("Hide tiny tiles below market value", 0, 10000, 0, 250)

    st.header("Market data")
    use_market_data = st.checkbox("Fetch live TA + valuation data", value=True)
    interval = st.selectbox("Primary technical candle size", ["1h", "4h", "1d", "1wk"], index=2)
    period = st.selectbox("Primary technical lookback", ["6mo", "1y", "2y", "5y"], index=1)
    build_mtf = st.checkbox("Build multi-timeframe trend matrix", value=True)
    benchmark = st.text_input("Benchmark", value="SPY")
    risk_free = st.number_input("Risk-free rate %", 0.0, 25.0, 4.5, 0.25) / 100

    st.header("Tax-loss harvesting")
    tlh_min_dollars = st.number_input("Min loss to flag ($)", 0, 100_000, 250, 50)
    tlh_min_pct = st.slider("Min loss to flag (%)", -50, 0, -5, 1) / 100

    st.header("Advanced analytics")
    enable_advanced = st.checkbox("Enable Performance / Projections / Factors", value=True)
    enable_options = st.checkbox("Enable Options screener (slower)", value=False)
    enable_valuation = st.checkbox("Enable advanced valuation (DCF, Graham, quality)", value=True)
    enable_macro = st.checkbox("Enable macro + insider/short data", value=True)
    mc_years = st.slider("Monte Carlo horizon (years)", 1, 30, 10)
    mc_annual_contrib = st.number_input("Annual contribution ($)", 0, 1_000_000, 0, 1000)
    bt_start_date = st.date_input("Backtest start date", value=pd.Timestamp("2020-01-01")).strftime("%Y-%m-%d")
    bt_rebalance = st.selectbox("Backtest rebalancing", ["none", "quarterly", "annually"], index=0)

    st.header("DCF model assumptions")
    dcf_growth = st.slider("Year 1-5 FCF growth %", 0, 30, 10) / 100
    dcf_discount = st.slider("Discount rate (WACC) %", 5, 15, 9) / 100

    st.header("Rebalance & tax")
    reb_band = st.slider("Target band tolerance (±%)", 1, 15, 5, 1) / 100
    st_tax_rate = st.slider("Short-term / ordinary rate %", 0, 50, 37, 1) / 100
    lt_tax_rate = st.slider("Long-term cap-gains rate %", 0, 40, 20, 1) / 100
    avoid_wash = st.checkbox("Avoid wash-sale loss lots", value=True)

    st.header("AI analyst")
    _ai_ready, _ai_msg = ai_analyst.availability()
    ai_model_label = st.selectbox("Model", list(ai_analyst.MODELS.keys()), index=1)
    ai_model = ai_analyst.MODELS[ai_model_label]
    st.caption("✅ AI commentary enabled." if _ai_ready else f"⚪ {_ai_msg}")


# =============================================================================
# Landing state (no file uploaded)
# =============================================================================
if uploaded is None:
    empty_state(
        "Upload your latest broker positions Excel file in the sidebar to generate the dashboard.",
        icon="📁",
    )
    st.markdown(
        """
        **Workflow:** download positions from your broker → upload here →
        review allocation, heat map, trend ratings, valuations, risk flags, income,
        tax lots, smart-money setups, tax-loss harvesting opportunities, and exports.

        **Privacy:** never commit `.xls`, `.xlsx`, or `.csv` broker files. The app
        starts empty and only analyzes the file you upload in the sidebar.
        """
    )
    st.stop()


# =============================================================================
# Parse + enrich
# =============================================================================
try:
    result = parse_broker_upload(uploaded)
except Exception as e:
    st.error(f"Could not parse this file: {e}")
    st.stop()

summary = result.summary.copy()
lots = result.lots.copy()
allocation = result.allocation.copy()

broker_total = result.broker_total if result.broker_total else allocation["Market Value"].sum()
ugl_total = summary["Unrealized P&L $"].fillna(0).sum()
today_total = summary["Today's Change $"].fillna(0).sum()
equity_count = summary.loc[summary["Asset Type"].eq("Stocks"), "Symbol"].nunique()
# Defensively exclude any income > 50% of market value (face-value-as-income errors)
_income_clean = summary.copy()
_income_clean["__ratio"] = _income_clean["Est. Annual Income"].fillna(0) / _income_clean["Market Value"].replace(0, np.nan)
_income_clean.loc[_income_clean["__ratio"] > 0.5, "Est. Annual Income"] = np.nan
income_total = _income_clean["Est. Annual Income"].fillna(0).sum()

securities_symbols = (
    summary[summary["Asset Type"].isin(["Stocks", "ETFs", "Mutual Funds"])]["Symbol"]
    .dropna().astype(str).str.upper().unique().tolist()
)
market_symbols = (
    summary[summary["Asset Type"].isin(["Stocks", "ETFs"])]["Symbol"]
    .dropna().astype(str).str.upper().unique().tolist()
)

# Live data enrichment
tech = pd.DataFrame()
mtf_tech = pd.DataFrame()
fund = pd.DataFrame()
returns = pd.DataFrame()
earnings = pd.DataFrame()

if use_market_data:
    with st.spinner("Fetching market data in parallel — TA + valuation + risk metrics..."):
        symbol_tuple = tuple(securities_symbols)
        market_tuple = tuple(market_symbols)
        tech = cached_technical_snapshot(symbol_tuple, period, interval)
        if build_mtf:
            mtf_plan = [("1h", "6mo"), ("4h", "1y"), ("1d", "2y"), ("1wk", "5y")]
            frames = []
            for tf, tf_period in mtf_plan:
                frame = cached_technical_snapshot(symbol_tuple, tf_period, tf)
                if not frame.empty:
                    frames.append(frame)
            mtf_tech = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        fund = cached_fundamentals(market_tuple)
        returns = cached_returns(tuple(market_symbols + ([benchmark] if benchmark else [])), "1y")
        earnings = cached_earnings(market_tuple)

# Merge enrichments onto summary
if not tech.empty:
    summary = summary.merge(tech.drop(columns=["Timeframe"], errors="ignore"), on="Symbol", how="left")
if not fund.empty:
    summary = summary.merge(fund, on="Symbol", how="left")

# Composite risk scores (uses everything merged above)
summary = compute_position_risk_scores(summary)

# Advisor flags
summary["Portfolio Weight"] = summary["Portfolio Weight"].fillna(0)
summary["Risk Flag"] = ""
summary.loc[summary["Portfolio Weight"] > 0.10, "Risk Flag"] += "Oversized position; "
if "Trend Rating" in summary.columns:
    summary.loc[summary["Trend Rating"].eq("Bearish") & (summary["Portfolio Weight"] > 0.01), "Risk Flag"] += "Bearish trend; "
if "Drawdown from 52W High %" in summary.columns:
    summary.loc[summary["Drawdown from 52W High %"] < -0.25, "Risk Flag"] += "Deep drawdown; "
summary["Risk Flag"] = summary["Risk Flag"].str.rstrip("; ")


# =============================================================================
# Top metrics row
# =============================================================================
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total portfolio value", fmt_currency(broker_total))
c2.metric("Unrealized P&L", fmt_currency(ugl_total), fmt_percent(ugl_total / broker_total, 2, signed=True) if broker_total else None)
c3.metric("Today's change", fmt_currency(today_total), fmt_percent(today_total / broker_total, 2, signed=True) if broker_total else None)
c4.metric("Equity positions", f"{equity_count} stocks")
c5.metric("Est. annual income", fmt_currency(income_total), fmt_percent(income_total / broker_total, 2) if broker_total else None)
c6.metric("Priced date", result.priced_date or "Not found")


# =============================================================================
# Portfolio grade scorecard
# =============================================================================
# Pre-compute the inputs we need for grading
_hhi = float((summary["Portfolio Weight"].fillna(0) ** 2).sum()) if not summary.empty else None
_cash_weight_for_grade = cash_weight if 'cash_weight' in dir() else None
_bull_weight = _bear_weight = None
if "Trend Rating" in summary.columns:
    _bull_weight = float(summary.loc[summary["Trend Rating"] == "Bullish", "Portfolio Weight"].sum())
    _bear_weight = float(summary.loc[summary["Trend Rating"] == "Bearish", "Portfolio Weight"].sum())

_sharpe = _max_dd_for_grade = None
if not returns.empty:
    _w_for_grade = summary.set_index("Symbol")["Portfolio Weight"].reindex(returns.columns).fillna(0)
    if _w_for_grade.sum() > 0:
        _port_ret = returns.mul(_w_for_grade, axis=1).sum(axis=1)
        _ann_vol = _port_ret.std() * np.sqrt(252)
        _ann_ret = _port_ret.mean() * 252
        _sharpe = (_ann_ret - risk_free) / _ann_vol if _ann_vol and pd.notna(_ann_vol) else None
        _max_dd_for_grade = max_drawdown_from_returns(_port_ret)

_tlh_for_grade = find_tlh_candidates(lots, min_loss_dollars=tlh_min_dollars, min_loss_pct=tlh_min_pct)
_tlh_unrealized = 0.0
if not _tlh_for_grade.empty and "TLH Candidate" in _tlh_for_grade.columns:
    _tlh_unrealized = float(_tlh_for_grade.loc[_tlh_for_grade["TLH Candidate"] == "✓", "Unrealized P&L $"].sum())

cash_weight_calc = float(allocation.loc[allocation["Asset Class"].eq("Cash"), "Market Value"].sum() / broker_total) if broker_total else 0

grade_result = portfolio_grade(
    summary,
    hhi=_hhi,
    sharpe=_sharpe,
    max_drawdown=_max_dd_for_grade,
    cash_weight=cash_weight_calc,
    bullish_weight=_bull_weight,
    bearish_weight=_bear_weight,
    tlh_unrealized_loss=_tlh_unrealized,
    portfolio_value=broker_total,
)

if grade_result.get("sub_scores"):
    grade_color = {"A+": "#22C55E", "A": "#22C55E", "A-": "#86EFAC", "B+": "#84CC16",
                   "B": "#FACC15", "B-": "#F59E0B", "C": "#F97316", "D": "#EF4444", "F": "#DC2626"}.get(
        grade_result["grade"], PALETTE["text_muted"])
    grade_letter = grade_result["grade"]
    grade_score = grade_result["score"]
    st.markdown(
        f"""
        <div class="epic-card" style="margin-top:1rem;display:flex;align-items:center;gap:24px;flex-wrap:wrap">
            <div style="text-align:center;min-width:120px">
                <div style="font-size:0.78rem;color:{PALETTE['text_muted']};letter-spacing:0.05em">PORTFOLIO GRADE</div>
                <div style="font-size:3.5rem;font-weight:800;color:{grade_color};line-height:1">{grade_letter}</div>
                <div style="color:{PALETTE['text_muted']};font-size:0.85rem">{grade_score:.1f} / 100</div>
            </div>
            <div style="flex:1;min-width:300px">
                {"".join([
                    f'<div style="margin-bottom:6px"><span style="display:inline-block;width:170px;font-size:0.82rem;color:{PALETTE["text_muted"]}">{k}</span>'
                    f'<span style="display:inline-block;width:80px;font-weight:600">{v:.0f}</span>'
                    f'<span style="color:{PALETTE["text_muted"]};font-size:0.78rem">{grade_result["rationale"].get(k, "")}</span></div>'
                    for k, v in grade_result["sub_scores"].items()
                ])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Advisor cockpit
# =============================================================================
flag_items: list[str] = []
cash_weight = float(allocation.loc[allocation["Asset Class"].eq("Cash"), "Market Value"].sum() / broker_total) if broker_total else 0
stock_weight = float(allocation.loc[allocation["Asset Class"].eq("Stocks"), "Market Value"].sum() / broker_total) if broker_total else 0
top1_weight = float(summary["Portfolio Weight"].max()) if not summary.empty else 0
top5_weight = float(summary.head(5)["Portfolio Weight"].sum()) if not summary.empty else 0

if cash_weight > 0.20:
    flag_items.append(f"Cash is high at {cash_weight:.1%}; decide whether it is dry powder or drag.")
if top1_weight > 0.12:
    flag_items.append(f"Largest holding is {top1_weight:.1%}; monitor single-name concentration.")
if top5_weight > 0.40:
    flag_items.append(f"Top 5 holdings are {top5_weight:.1%}; portfolio is concentrated.")
if "Trend Rating" in summary.columns:
    bearish_weight = summary.loc[summary["Trend Rating"].eq("Bearish"), "Portfolio Weight"].sum()
    if bearish_weight > 0.15:
        flag_items.append(f"Bearish technical exposure is {bearish_weight:.1%} on EMA10/EMA20/EMA50 + 200 MA structure.")
if "Risk Tier" in summary.columns:
    high_risk_weight = summary.loc[summary["Risk Tier"] == "High", "Portfolio Weight"].sum()
    if high_risk_weight > 0.15:
        flag_items.append(f"High-risk positions account for {high_risk_weight:.1%} of the portfolio.")
if not flag_items:
    flag_items.append("No major dashboard-level flags triggered by the current rules.")

with st.expander("🎯 Advisor cockpit — what needs attention first", expanded=True):
    for item in flag_items:
        st.write("• " + item)

st.divider()


# =============================================================================
# Main visuals: allocation + heat map
# =============================================================================
left, right = st.columns([1.05, 2.15])
with left:
    section_header("Asset allocation")
    fig_alloc = px.pie(allocation, names="Asset Class", values="Market Value", hole=0.52,
                      color_discrete_sequence=px.colors.qualitative.Set2)
    fig_alloc.update_traces(textposition="inside", textinfo="percent+label")
    fig_alloc.update_layout(margin=dict(l=0, r=0, t=20, b=20), height=380,
                           template="plotly_dark", showlegend=True)
    st.plotly_chart(fig_alloc, use_container_width=True)

    section_header("Allocation bar")
    fig_bar = px.bar(allocation.sort_values("Market Value", ascending=True),
                     x="Market Value", y="Asset Class", orientation="h", text="Weight",
                     color="Asset Class", color_discrete_sequence=px.colors.qualitative.Set2)
    fig_bar.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig_bar.update_layout(height=260, margin=dict(l=0, r=20, t=10, b=10),
                         showlegend=False, template="plotly_dark")
    st.plotly_chart(fig_bar, use_container_width=True)

with right:
    section_header("Portfolio heat map", "Colored by your sidebar selection")
    hm = summary[summary["Market Value"].fillna(0) >= min_tile].copy()
    if hm.empty:
        empty_state("No holdings passed the market value filter.")
    else:
        hover_cols = {
            "Description": True, "Market Value": ":$,.2f",
            "Portfolio Weight": ":.2%", "Unrealized P&L $": ":$,.2f",
            "Unrealized P&L %": ":.2%", "Today's Change $": ":$,.2f",
            "Today's Change %": ":.2%",
        }
        if "Trend Rating" in hm.columns:
            hover_cols.update({"Trend Rating": True, "Trend Setup": True, "Risk Tier": True})

        color_col = color_by
        if color_col not in hm.columns:
            color_col = "Portfolio Weight"

        fig = px.treemap(
            hm, path=["Asset Type", "Symbol"], values="Market Value",
            color=color_col, color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0 if color_col in ("Today's Change %", "Unrealized P&L %") else hm[color_col].median(),
            hover_data=hover_cols,
        )
        fig.update_traces(texttemplate="<b>%{label}</b><br>%{value:$,.0f}")
        fig.update_layout(margin=dict(l=0, r=0, t=20, b=20), height=650, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Tabs
# =============================================================================
labels = [
    "🚦 Action Center",
    "📈 Holdings", "📊 Trend Ratings", "💰 Valuation", "⚠️ Risk",
    "🏆 Winners/Losers", "💵 Income", "📋 Tax Lots", "♻️ Rebalance",
    "💎 Smart Money", "🧾 Tax Loss Harvest", "📰 News & Earnings",
    "🎯 Performance", "🎲 Projections", "📞 Options", "🔬 Factors",
    "🌍 Macro", "🧬 Themes", "🤖 AI Analyst", "📤 Exports",
]
tabs = st.tabs(labels)
tab_action, tab_holdings, tab_trend, tab_val, tab_risk, tab_wl, tab_inc, tab_lots, \
    tab_reb, tab_sm, tab_tlh, tab_news, tab_perf, tab_proj, tab_options, tab_factors, \
    tab_macro, tab_themes, tab_ai, tab_exp = tabs

# Shared rebalance targets (editable in the Rebalance tab; read everywhere).
DEFAULT_TARGETS = {"Stocks": 0.35, "ETFs": 0.15, "Mutual Funds": 0.25,
                   "Cash": 0.20, "Fixed Income": 0.05}
targets = st.session_state.get("reb_targets", DEFAULT_TARGETS)


# -----------------------------------------------------------------------------
# Tab: Action Center  (NEW — cross-module prioritized to-do list)
# -----------------------------------------------------------------------------
with tab_action:
    section_header(
        "Action Center",
        "Everything that needs your attention today, scanned across the whole book "
        "and ranked by urgency.",
    )

    _earnings_df = earnings if not earnings.empty else None
    _iv_df = st.session_state.get("iv_df")
    _tlh_lots = find_tlh_candidates(lots, min_loss_dollars=tlh_min_dollars,
                                    min_loss_pct=tlh_min_pct) if not lots.empty else None

    sig = build_signals(
        summary, allocation, lots,
        broker_total=broker_total, targets=targets, band_tolerance=reb_band,
        tlh_lots=_tlh_lots, earnings=_earnings_df, iv_table=_iv_df,
    )
    counts = signal_counts(sig)

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("🔴 Urgent", counts["🔴"])
    a2.metric("🟡 Review", counts["🟡"])
    a3.metric("🟢 Opportunistic", counts["🟢"])
    a4.metric("Total signals", counts["total"])

    if sig.empty:
        empty_state("All clear — nothing is breaching your thresholds right now.", icon="✅")
    else:
        st.dataframe(sig, use_container_width=True, hide_index=True)
        st.caption(
            "Signals are derived from your live holdings, target bands, and tax lots. "
            "Earnings and IV signals populate once you open those tabs in this session."
        )

    st.divider()
    if st.button("🤖 Have the AI analyst brief me on the book", use_container_width=True):
        with st.spinner("Analyzing…"):
            st.markdown(ai_analyst.portfolio_commentary(
                summary, allocation, broker_total=broker_total, ugl_total=ugl_total,
                hhi=_hhi, model=ai_model,
            ))





# -----------------------------------------------------------------------------
# Tab: Holdings
# -----------------------------------------------------------------------------
with tab_holdings:
    section_header("Cleaned holdings summary",
                   "All positions with weights, P&L, trend, and composite risk score")

    # Filters
    hf1, hf2, hf3 = st.columns([2, 2, 1])
    with hf1:
        asset_filter = st.multiselect(
            "Asset type filter",
            options=sorted(summary["Asset Type"].dropna().unique().tolist()),
            default=sorted(summary["Asset Type"].dropna().unique().tolist()),
        )
    with hf2:
        if "Trend Rating" in summary.columns:
            trend_filter = st.multiselect(
                "Trend filter",
                options=["Bullish", "Neutral", "Bearish", "No data"],
                default=["Bullish", "Neutral", "Bearish", "No data"],
            )
        else:
            trend_filter = None
    with hf3:
        hide_bonds = st.checkbox("Hide bonds", value=False)

    filtered = summary.copy()
    if asset_filter:
        filtered = filtered[filtered["Asset Type"].isin(asset_filter)]
    if trend_filter is not None and "Trend Rating" in filtered.columns:
        filtered = filtered[filtered["Trend Rating"].isin(trend_filter)]
    if hide_bonds and "Is Bond" in filtered.columns:
        filtered = filtered[~filtered["Is Bond"].fillna(False).astype(bool)]

    base_cols = ["Symbol", "Description", "Asset Type", "Shares", "Last Price",
                 "Market Value", "Portfolio Weight", "Total Cost", "Avg Cost",
                 "Unrealized P&L $", "Unrealized P&L %", "Today's Change $",
                 "Today's Change %", "Est. Annual Income", "Yield on MV",
                 "Lot Count", "Risk Score", "Risk Tier", "Risk Flag"]
    extra_cols = [c for c in ["Trend Rating", "Sector", "Industry", "Beta",
                              "Trailing P/E", "Forward P/E"] if c in filtered.columns]
    display_df = filtered[[c for c in base_cols + extra_cols if c in filtered.columns]]
    st.caption(f"Showing {len(display_df)} of {len(summary)} positions.")
    st.dataframe(
        style_cells(display_df, trend_cols=["Trend Rating"], risk_cols=["Risk Tier"]),
        use_container_width=True, hide_index=True,
    )


# -----------------------------------------------------------------------------
# Tab: Trend Ratings
# -----------------------------------------------------------------------------
with tab_trend:
    section_header(
        "EMA + 200 MA trend rating engine",
        "Bullish = EMA10 > EMA20 > EMA50 and Close > 200 MA · Bearish = inverse · "
        "everything else is Neutral.",
    )
    st.caption(
        f"Primary table uses **{interval}** candles. Multi-timeframe matrix below "
        "shows hourly / 4H / daily / weekly side-by-side."
    )
    if tech.empty:
        empty_state("Turn on market data in the sidebar to calculate live technical ratings.")
    else:
        tech_view = summary[[c for c in [
            "Symbol", "Description", "Asset Type", "Market Value", "Portfolio Weight",
            "Trend Rating", "Trend Setup", "Last Close", "EMA 10", "EMA 20",
            "EMA 50", "MA 200", "Distance from 50 EMA %", "Distance from 200 MA %",
            "Drawdown from 52W High %",
        ] if c in summary.columns]].copy()
        rating_counts = tech_view.groupby("Trend Rating", dropna=False)["Portfolio Weight"].sum().reset_index()

        colA, colB = st.columns([1, 2])
        with colA:
            st.metric(f"Bullish exposure ({interval})", fmt_percent(rating_counts.loc[rating_counts['Trend Rating'].eq('Bullish'), 'Portfolio Weight'].sum(), 1))
            st.metric(f"Bearish exposure ({interval})", fmt_percent(rating_counts.loc[rating_counts['Trend Rating'].eq('Bearish'), 'Portfolio Weight'].sum(), 1))
            st.metric(f"Neutral / no-data ({interval})", fmt_percent(rating_counts.loc[~rating_counts['Trend Rating'].isin(['Bullish','Bearish']), 'Portfolio Weight'].sum(), 1))
        with colB:
            color_map = {"Bullish": PALETTE["bullish"], "Bearish": PALETTE["bearish"], "Neutral": PALETTE["neutral"]}
            fig_rating = px.bar(rating_counts, x="Trend Rating", y="Portfolio Weight",
                              text="Portfolio Weight", color="Trend Rating",
                              color_discrete_map=color_map,
                              title=f"Portfolio exposure by trend — {interval} candles")
            fig_rating.update_traces(texttemplate="%{text:.1%}")
            fig_rating.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=20),
                                    showlegend=False, template="plotly_dark")
            st.plotly_chart(fig_rating, use_container_width=True)

        if not mtf_tech.empty:
            st.markdown("### Multi-timeframe trend matrix")
            mtf = mtf_tech.merge(
                summary[["Symbol", "Description", "Asset Type", "Market Value", "Portfolio Weight"]],
                on="Symbol", how="left",
            )
            tf_order = ["1h", "4h", "1d", "1wk"]
            matrix = mtf.pivot_table(
                index=["Symbol", "Description", "Asset Type", "Portfolio Weight"],
                columns="Timeframe", values="Trend Rating", aggfunc="first",
            ).reset_index()
            for tf in tf_order:
                if tf not in matrix.columns:
                    matrix[tf] = "No data"
            matrix["Alignment Score"] = matrix[tf_order].apply(
                lambda r: sum(1 if x == "Bullish" else -1 if x == "Bearish" else 0 for x in r), axis=1
            )
            matrix["Read"] = np.select(
                [matrix["Alignment Score"].ge(3),
                 matrix["Alignment Score"].le(-3),
                 matrix[["1d", "1wk"]].eq("Bullish").all(axis=1),
                 matrix[["1d", "1wk"]].eq("Bearish").all(axis=1)],
                ["Bullish across most timeframes",
                 "Bearish across most timeframes",
                 "Higher-timeframe bullish",
                 "Higher-timeframe bearish"],
                default="Mixed / timeframe conflict",
            )
            display_cols = ["Symbol", "Description", "Asset Type", "Portfolio Weight"] + tf_order + ["Alignment Score", "Read"]
            st.dataframe(
                style_cells(
                    matrix[display_cols].sort_values(["Alignment Score", "Portfolio Weight"], ascending=[False, False]),
                    trend_cols=tf_order,
                ),
                use_container_width=True, hide_index=True,
            )

            mtf_exposure = mtf.groupby(["Timeframe", "Trend Rating"], dropna=False)["Portfolio Weight"].sum().reset_index()
            fig_mtf = px.bar(mtf_exposure, x="Timeframe", y="Portfolio Weight",
                            color="Trend Rating", text="Portfolio Weight",
                            category_orders={"Timeframe": tf_order},
                            color_discrete_map=color_map,
                            title="Bullish / bearish / neutral exposure by timeframe")
            fig_mtf.update_traces(texttemplate="%{text:.1%}")
            fig_mtf.update_layout(height=360, margin=dict(l=0, r=0, t=45, b=20), template="plotly_dark")
            st.plotly_chart(fig_mtf, use_container_width=True)

            st.markdown("### Detailed multi-timeframe EMA table")
            detailed_cols = ["Symbol", "Timeframe", "Trend Rating", "Trend Setup",
                            "Last Close", "EMA 10", "EMA 20", "EMA 50", "MA 200",
                            "Distance from 200 MA %", "Drawdown from 52W High %"]
            st.dataframe(
                mtf[[c for c in detailed_cols if c in mtf.columns]].sort_values(["Symbol", "Timeframe"]),
                use_container_width=True, hide_index=True,
            )

        st.markdown("### Primary timeframe details")
        st.dataframe(
            tech_view.sort_values(["Trend Rating", "Portfolio Weight"], ascending=[True, False]),
            use_container_width=True, hide_index=True,
        )


# -----------------------------------------------------------------------------
# Tab: Valuation (UPGRADED — DCF, Graham, PEG, quality)
# -----------------------------------------------------------------------------
with tab_val:
    section_header("Valuation + fundamentals + quality screen",
                   "DCF, Graham Number, PEG, owner-earnings yield, and a composite verdict per stock.")

    if fund.empty:
        empty_state("Turn on market data in the sidebar to pull valuation/fundamental data.")
    else:
        # ---- Sector treemap (filtered to exclude bonds + 'None' sectors) ----
        # The bond holdings show "None" sector from yfinance and were dominating the chart
        sector_summary = summary[
            (~summary.get("Is Bond", pd.Series(False, index=summary.index)).fillna(False).astype(bool))
            & (summary["Asset Type"].isin(["Stocks", "ETFs"]))
        ]
        sector_df = sector_concentration_hhi(sector_summary)
        if not sector_df.empty:
            colA, colB = st.columns([2, 1])
            with colA:
                fig_sector = px.treemap(
                    sector_df, path=["Sector"], values="Market Value",
                    color="Portfolio Weight", color_continuous_scale="Tealgrn",
                    title="Sector exposure (equities + ETFs only)",
                )
                fig_sector.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=10),
                                        template="plotly_dark")
                st.plotly_chart(fig_sector, use_container_width=True)
            with colB:
                hhi_sec = float((sector_df["Portfolio Weight"] ** 2).sum())
                eff_sectors = 1 / hhi_sec if hhi_sec else np.nan
                st.metric("Sector HHI", f"{hhi_sec:.3f}",
                        help="0 = perfectly diversified, 1 = single-sector. <0.20 is typically well-diversified.")
                st.metric("Effective # of sectors", f"{eff_sectors:.1f}" if pd.notna(eff_sectors) else "N/A")
                st.dataframe(
                    sector_df[["Sector", "Portfolio Weight", "Positions"]].head(10),
                    use_container_width=True, hide_index=True,
                )

        st.divider()

        # ---- Quick fundamentals table (legacy) ----
        st.markdown("### Quick valuation snapshot")
        vcols = ["Symbol", "Description", "Sector", "Industry", "Market Value",
                 "Portfolio Weight", "Market Cap", "Trailing P/E", "Forward P/E",
                 "Price/Sales", "Price/Book", "EV/EBITDA", "Dividend Yield",
                 "Beta", "Analyst Target Mean", "Recommendation", "Market Data Status"]
        val = summary[[c for c in vcols if c in summary.columns]].copy()
        # Don't show bonds/MFs here, they don't have meaningful fundamentals
        val = val[~summary.get("Is Bond", pd.Series(False, index=summary.index)).fillna(False).astype(bool)]
        st.dataframe(val.sort_values("Portfolio Weight", ascending=False),
                    use_container_width=True, hide_index=True)

        # ---- ADVANCED: DCF + Graham + Quality (only for stocks) ----
        if enable_valuation:
            st.divider()
            section_header("Advanced valuation — DCF, Graham Number, quality score",
                          f"Assumptions: {dcf_growth:.0%} initial growth, {dcf_discount:.0%} discount rate. Adjust in sidebar.")
            stock_symbols = summary[
                (summary["Asset Type"] == "Stocks")
                & (~summary.get("Is Bond", pd.Series(False, index=summary.index)).fillna(False).astype(bool))
            ]["Symbol"].dropna().tolist()

            if not stock_symbols:
                st.info("No individual stock holdings to value (ETFs and bonds are excluded).")
            else:
                with st.spinner(f"Running DCF + valuation models on {len(stock_symbols)} stocks..."):
                    val_df = cached_advanced_valuations(tuple(stock_symbols), dcf_growth, dcf_discount)

                if val_df.empty:
                    st.warning("Could not retrieve valuation data.")
                else:
                    # Defensive: guarantee the columns we read below exist
                    for _col, _default in [
                        ("Valuation Verdict", "—"), ("Quality Score", np.nan),
                        ("Quality Tier", ""), ("DCF Upside %", np.nan),
                        ("Trailing P/E", np.nan), ("PEG (calc)", np.nan),
                        ("ROE", np.nan), ("Sector", ""),
                    ]:
                        if _col not in val_df.columns:
                            val_df[_col] = _default

                    # Top-level metrics
                    n_undervalued = int((val_df["Valuation Verdict"].isin(["Strongly undervalued", "Undervalued"])).sum())
                    n_overvalued = int((val_df["Valuation Verdict"].isin(["Strongly overvalued", "Overvalued"])).sum())
                    avg_quality = val_df["Quality Score"].mean()

                    vm1, vm2, vm3, vm4 = st.columns(4)
                    vm1.metric("Undervalued holdings", str(n_undervalued))
                    vm2.metric("Overvalued holdings", str(n_overvalued))
                    vm3.metric("Average quality score", f"{avg_quality:.1f}" if pd.notna(avg_quality) else "—")
                    vm4.metric("Stocks analyzed", str(len(val_df)))

                    # Warn if the data came back sparse (yfinance rate-limited)
                    n_with_dcf = int(val_df["DCF Upside %"].notna().sum())
                    if n_with_dcf == 0:
                        st.info(
                            "Valuation models couldn't compute for any holding right now — "
                            "this usually means yfinance is rate-limiting fundamental data. "
                            "Wait a few minutes and reload, or uncheck/recheck 'Enable advanced "
                            "valuation' in the sidebar to refresh."
                        )

                    # Headline scatter: DCF upside vs Quality
                    scatter_df = val_df.dropna(subset=["DCF Upside %", "Quality Score"]).copy()
                    if not scatter_df.empty:
                        # Merge in portfolio weight for bubble size
                        weight_map = summary.set_index("Symbol")["Portfolio Weight"].to_dict()
                        scatter_df["Portfolio Weight"] = scatter_df["Symbol"].map(weight_map).fillna(0.01)
                        st.markdown("#### Quality vs DCF Upside (bubble size = portfolio weight)")
                        fig_qv = px.scatter(
                            scatter_df, x="Quality Score", y="DCF Upside %",
                            size=scatter_df["Portfolio Weight"] * 1000 + 5,
                            color="Valuation Verdict", text="Symbol",
                            hover_data=["Trailing P/E", "PEG (calc)", "ROE", "Sector"],
                            color_discrete_map={
                                "Strongly undervalued": PALETTE["bullish"],
                                "Undervalued": "#86EFAC",
                                "Fairly valued": PALETTE["neutral"],
                                "Overvalued": "#FCA5A5",
                                "Strongly overvalued": PALETTE["bearish"],
                            },
                        )
                        fig_qv.update_traces(textposition="top center")
                        fig_qv.add_hline(y=0, line_dash="dot", line_color=PALETTE["text_muted"])
                        fig_qv.add_vline(x=55, line_dash="dot", line_color=PALETTE["text_muted"])
                        fig_qv.update_layout(height=540, margin=dict(l=0, r=0, t=20, b=10),
                                            template="plotly_dark", yaxis_tickformat=".0%",
                                            xaxis_title="Quality Score (0-100)",
                                            yaxis_title="DCF Upside / Downside",
                                            legend_orientation="h")
                        st.plotly_chart(fig_qv, use_container_width=True)
                        st.caption(
                            "**Top-right quadrant = high quality + cheap = best opportunities.** "
                            "Top-left = cheap junk. Bottom-right = expensive quality. "
                            "Bottom-left = expensive low-quality (worst)."
                        )

                    # Detailed table
                    st.markdown("#### Per-stock valuation details")
                    display_cols = [
                        "Symbol", "Sector", "Price", "Market Cap",
                        "Trailing P/E", "Forward P/E", "PEG (calc)", "PEG Verdict",
                        "DCF Fair Value", "DCF Upside %", "Implied Growth (Reverse-DCF)",
                        "Graham Number", "Graham Upside %",
                        "Owner Earnings Yield", "ROE", "Quality Score", "Quality Tier",
                        "Analyst Target", "Analyst Upside %", "Valuation Verdict",
                    ]
                    display_df = val_df[[c for c in display_cols if c in val_df.columns]].copy()

                    # Pretty formatting
                    def _fmt_pct(x):
                        return f"{x:+.1%}" if pd.notna(x) else ""
                    def _fmt_dol(x):
                        return f"${x:,.2f}" if pd.notna(x) and abs(x) < 1e10 else ""
                    def _fmt_num(x, d=2):
                        return f"{x:.{d}f}" if pd.notna(x) else ""

                    for col in ["DCF Upside %", "Graham Upside %", "Implied Growth (Reverse-DCF)",
                               "Owner Earnings Yield", "ROE", "Analyst Upside %"]:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].map(_fmt_pct)
                    for col in ["Price", "DCF Fair Value", "Graham Number", "Analyst Target"]:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].map(_fmt_dol)
                    for col in ["Trailing P/E", "Forward P/E", "PEG (calc)"]:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].map(lambda x: _fmt_num(x, 2))
                    if "Market Cap" in display_df.columns:
                        display_df["Market Cap"] = display_df["Market Cap"].map(
                            lambda x: f"${x/1e9:.1f}B" if pd.notna(x) and x > 0 else "")

                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                    # Best opportunities call-out
                    best_opps = val_df[
                        (val_df["Valuation Verdict"].isin(["Strongly undervalued", "Undervalued"]))
                        & (val_df["Quality Tier"].isin(["Good", "Excellent"]))
                    ].sort_values("DCF Upside %", ascending=False)
                    if not best_opps.empty:
                        with st.expander(f"⭐ Top picks: quality + undervalued ({len(best_opps)} stocks)"):
                            st.dataframe(
                                best_opps[["Symbol", "Sector", "DCF Upside %", "Quality Score",
                                          "Trailing P/E", "PEG (calc)", "Valuation Verdict"]],
                                use_container_width=True, hide_index=True,
                            )

                    # Worst — overvalued + low quality
                    worst = val_df[
                        (val_df["Valuation Verdict"].isin(["Strongly overvalued", "Overvalued"]))
                        & (val_df["Quality Tier"].isin(["Low", "Average"]))
                    ].sort_values("DCF Upside %", ascending=True)
                    if not worst.empty:
                        with st.expander(f"⚠️ Watch list: overvalued + lower quality ({len(worst)} stocks)"):
                            st.dataframe(
                                worst[["Symbol", "Sector", "DCF Upside %", "Quality Score",
                                      "Trailing P/E", "PEG (calc)", "Valuation Verdict"]],
                                use_container_width=True, hide_index=True,
                            )

                    # Single-stock deep dive
                    st.markdown("#### Single-stock multiple expansion analysis")
                    pe_sym = st.selectbox("Pick a stock for historical P/E band analysis",
                                         val_df["Symbol"].tolist())
                    if pe_sym:
                        with st.spinner(f"Loading {pe_sym} historical P/E..."):
                            pe_band = cached_pe_band(pe_sym)
                        if pe_band and pe_band.get("avg_pe"):
                            pb1, pb2, pb3, pb4 = st.columns(4)
                            pb1.metric("Current P/E", f"{pe_band.get('current_pe', 0):.1f}" if pd.notna(pe_band.get('current_pe')) else "—")
                            pb2.metric(f"{pe_band['n_years']}-year avg P/E", f"{pe_band['avg_pe']:.1f}")
                            pb3.metric("Min / Max", f"{pe_band['min_pe']:.1f} / {pe_band['max_pe']:.1f}")
                            cva = pe_band.get("current_vs_avg")
                            pb4.metric("vs historical avg", _fmt_pct(cva) if pd.notna(cva) else "—",
                                      "expensive" if (cva or 0) > 0.10 else ("cheap" if (cva or 0) < -0.10 else "in line"))
                        else:
                            st.caption(f"Not enough historical earnings data for {pe_sym}.")


# -----------------------------------------------------------------------------
# Tab: Risk
# -----------------------------------------------------------------------------
with tab_risk:
    section_header("Portfolio risk dashboard")
    hhi = float((summary["Portfolio Weight"].fillna(0) ** 2).sum())
    eff_n = 1 / hhi if hhi else np.nan
    beta_weighted = np.nan
    if "Beta" in summary.columns:
        beta_weighted = (summary["Beta"].fillna(0) * summary["Portfolio Weight"].fillna(0)).sum()

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Largest position", fmt_percent(top1_weight, 1))
    r2.metric("Top 5 weight", fmt_percent(top5_weight, 1))
    r3.metric("Effective # holdings", f"{eff_n:.1f}" if pd.notna(eff_n) else "N/A")
    r4.metric("Weighted beta", f"{beta_weighted:.2f}" if pd.notna(beta_weighted) else "N/A")

    if returns.empty:
        empty_state("Turn on market data to estimate volatility, drawdown, correlations, and benchmark stats.")
    else:
        weights = summary.set_index("Symbol")["Portfolio Weight"].reindex(returns.columns).fillna(0)
        if weights.sum() > 0:
            port_ret = returns.mul(weights, axis=1).sum(axis=1)
            ann_vol = port_ret.std() * np.sqrt(252)
            ann_ret = port_ret.mean() * 252
            sharpe = (ann_ret - risk_free) / ann_vol if ann_vol and pd.notna(ann_vol) else np.nan
            dd = max_drawdown_from_returns(port_ret)
            rr1, rr2, rr3, rr4 = st.columns(4)
            rr1.metric("1Y est. return", fmt_percent(ann_ret, 1))
            rr2.metric("1Y est. volatility", fmt_percent(ann_vol, 1))
            rr3.metric("Est. Sharpe", f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A")
            rr4.metric("Max drawdown (1Y)", fmt_percent(dd, 1) if pd.notna(dd) else "N/A")

            # NEW: benchmark comparison
            if benchmark and benchmark in returns.columns:
                st.markdown("### Portfolio vs benchmark")
                bcmp = benchmark_comparison(returns, weights, benchmark=benchmark)
                if not bcmp.empty:
                    fig_bench = go.Figure()
                    fig_bench.add_trace(go.Scatter(
                        x=bcmp.index, y=bcmp["Portfolio"] * 100, name="Portfolio",
                        line=dict(width=2, color=PALETTE["primary"]),
                    ))
                    if benchmark in bcmp.columns:
                        fig_bench.add_trace(go.Scatter(
                            x=bcmp.index, y=bcmp[benchmark] * 100, name=benchmark,
                            line=dict(width=2, color=PALETTE["info"], dash="dot"),
                        ))
                    fig_bench.update_layout(
                        height=380, margin=dict(l=0, r=0, t=20, b=10),
                        yaxis_title="Cumulative return (%)", template="plotly_dark",
                        legend_orientation="h",
                    )
                    st.plotly_chart(fig_bench, use_container_width=True)

                    bstats = benchmark_stats(returns, weights, benchmark=benchmark, risk_free=risk_free)
                    if bstats:
                        b1, b2, b3, b4 = st.columns(4)
                        b1.metric("Jensen's alpha", fmt_percent(bstats["Alpha (Jensen)"], 2, signed=True))
                        b2.metric(f"Beta vs {benchmark}", f"{bstats['Beta vs benchmark']:.2f}")
                        b3.metric("Tracking error", fmt_percent(bstats["Tracking error"], 2))
                        b4.metric("Information ratio", f"{bstats['Information ratio']:.2f}")

            # Drawdown chart
            st.markdown("### Portfolio drawdown")
            dds = drawdown_series(port_ret)
            if not dds.empty:
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=dds.index, y=dds * 100, fill="tozeroy", name="Drawdown",
                    line=dict(color=PALETTE["bearish"], width=1),
                    fillcolor="rgba(239,68,68,0.15)",
                ))
                fig_dd.update_layout(
                    height=280, margin=dict(l=0, r=0, t=10, b=10),
                    yaxis_title="Drawdown (%)", template="plotly_dark", showlegend=False,
                )
                st.plotly_chart(fig_dd, use_container_width=True)

        # Correlation matrix
        st.markdown("### 1-year return correlation matrix")
        corr = returns[[c for c in returns.columns if c != benchmark]].corr()
        if corr.shape[0] >= 2:
            fig_corr = px.imshow(
                corr, text_auto=False, aspect="auto",
                color_continuous_scale="RdBu", zmin=-1, zmax=1,
            )
            fig_corr.update_layout(height=620, margin=dict(l=0, r=0, t=20, b=0),
                                  template="plotly_dark")
            st.plotly_chart(fig_corr, use_container_width=True)

            # Correlation clusters
            clusters = correlation_clusters(returns[[c for c in returns.columns if c != benchmark]], threshold=0.8)
            if not clusters.empty:
                with st.expander(f"⚡ Highly-correlated pairs (ρ ≥ 0.80) — {len(clusters)} pairs"):
                    st.dataframe(clusters, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Tab: Winners / Losers
# -----------------------------------------------------------------------------
with tab_wl:
    c1, c2 = st.columns(2)
    with c1:
        section_header("Biggest unrealized winners")
        st.dataframe(
            summary.sort_values("Unrealized P&L $", ascending=False).head(15)[
                ["Symbol", "Market Value", "Unrealized P&L $", "Unrealized P&L %", "Portfolio Weight"]
            ],
            use_container_width=True, hide_index=True,
        )
    with c2:
        section_header("Biggest unrealized losers")
        st.dataframe(
            summary.sort_values("Unrealized P&L $", ascending=True).head(15)[
                ["Symbol", "Market Value", "Unrealized P&L $", "Unrealized P&L %", "Portfolio Weight"]
            ],
            use_container_width=True, hide_index=True,
        )
    section_header("Today's movers")
    st.dataframe(
        summary.sort_values("Today's Change $", ascending=False)[
            ["Symbol", "Market Value", "Today's Change $", "Today's Change %", "Portfolio Weight"]
        ].head(20),
        use_container_width=True, hide_index=True,
    )


# -----------------------------------------------------------------------------
# Tab: Income
# -----------------------------------------------------------------------------
with tab_inc:
    section_header("Income dashboard", "Separates bond coupons from equity dividends so the numbers actually mean something.")

    # Defensive: ensure Is Bond column exists even on old parser state
    if "Is Bond" not in summary.columns:
        summary["Is Bond"] = False

    income_df = summary[summary["Est. Annual Income"].fillna(0) > 0].copy()

    if income_df.empty:
        empty_state("No income-paying positions found in the export.")
    else:
        # Split by asset class
        equity_income = income_df[~income_df["Is Bond"].fillna(False).astype(bool)].copy()
        bond_income = income_df[income_df["Is Bond"].fillna(False).astype(bool)].copy()

        equity_income_total = float(equity_income["Est. Annual Income"].sum())
        bond_income_total = float(bond_income["Est. Annual Income"].sum())
        grand_total = equity_income_total + bond_income_total

        # Sanity-check: if any single position's "income" is > 50% of its MV,
        # it's probably bad data — surface a warning rather than silently breaking.
        ratio = income_df["Est. Annual Income"] / income_df["Market Value"].replace(0, np.nan)
        suspicious = income_df[ratio > 0.5][["Symbol", "Description", "Market Value", "Est. Annual Income"]]
        if not suspicious.empty:
            with st.expander(f"⚠️ Suspicious income values ({len(suspicious)}) — likely face value or principal mistakenly labelled as income"):
                st.dataframe(suspicious, use_container_width=True, hide_index=True)
                st.caption("These are excluded from the headline totals below.")
                # Strip suspicious rows from totals
                clean_mask = ratio <= 0.5
                income_df = income_df[clean_mask]
                equity_income = equity_income[equity_income["Symbol"].isin(income_df["Symbol"])]
                bond_income = bond_income[bond_income["Symbol"].isin(income_df["Symbol"])]
                equity_income_total = float(equity_income["Est. Annual Income"].sum())
                bond_income_total = float(bond_income["Est. Annual Income"].sum())
                grand_total = equity_income_total + bond_income_total

        # Headline metrics
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.metric("Total annual income (corrected)", fmt_currency(grand_total))
        ic2.metric("From equities (dividends)", fmt_currency(equity_income_total))
        ic3.metric("From bonds (coupons)", fmt_currency(bond_income_total))
        ic4.metric("Portfolio yield", fmt_percent(grand_total / broker_total, 2) if broker_total else "—")

        # Monthly projection
        st.markdown("### Monthly income projection")
        monthly = grand_total / 12
        st.caption(f"Estimated monthly income: **{fmt_currency(monthly)}** (assumes even distribution; "
                  f"actual months vary based on ex-div dates).")

        # Equity income chart
        if not equity_income.empty:
            st.markdown("### Equity dividend income")
            top_eq = equity_income.nlargest(25, "Est. Annual Income")
            fig_eq = px.bar(
                top_eq, x="Symbol", y="Est. Annual Income",
                hover_data=["Description", "Market Value", "Yield on MV"],
                color="Yield on MV", color_continuous_scale="Tealgrn",
            )
            fig_eq.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=20),
                                template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_eq, use_container_width=True)
            st.dataframe(
                equity_income.sort_values("Est. Annual Income", ascending=False)[
                    ["Symbol", "Description", "Asset Type", "Market Value",
                     "Est. Annual Income", "Yield on MV", "Portfolio Weight"]
                ],
                use_container_width=True, hide_index=True,
            )

        # Bond coupon income (separately)
        if not bond_income.empty:
            st.markdown("### Bond coupon income")
            bond_display = bond_income.sort_values("Est. Annual Income", ascending=False)[
                ["Symbol", "Description", "Market Value", "Coupon Rate",
                 "Est. Annual Income", "Yield on MV", "Portfolio Weight"]
            ].copy()
            st.dataframe(bond_display, use_container_width=True, hide_index=True)
            st.caption(
                "Bond income shown here is the coupon (interest) payment, not the principal. "
                "Coupon rate is parsed from the description when possible — if it can't be detected, "
                "the row is flagged. Most coupon payments are made semi-annually."
            )

        # Income concentration
        st.markdown("### Income concentration")
        all_inc = pd.concat([equity_income, bond_income], ignore_index=True)
        if not all_inc.empty:
            top5_inc = float(all_inc.nlargest(5, "Est. Annual Income")["Est. Annual Income"].sum())
            top10_inc = float(all_inc.nlargest(10, "Est. Annual Income")["Est. Annual Income"].sum())
            cc1, cc2 = st.columns(2)
            cc1.metric("Top 5 % of total income", fmt_percent(top5_inc / grand_total, 1) if grand_total else "—")
            cc2.metric("Top 10 % of total income", fmt_percent(top10_inc / grand_total, 1) if grand_total else "—")


# -----------------------------------------------------------------------------
# Tab: Tax Lots
# -----------------------------------------------------------------------------
with tab_lots:
    section_header(
        "Tax-lot / source-row cleanup",
        "Parser avoids double-counting broker summary rows and keeps usable lot rows for trade-date and basis analysis.",
    )
    if lots.empty:
        empty_state("No lot rows found.")
    else:
        st.dataframe(lots, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Tab: Rebalance
# -----------------------------------------------------------------------------
with tab_reb:
    section_header(
        "Rebalance & tax-aware trade generator",
        "Set target weights, then generate the exact lots to sell and buys to make — "
        "harvesting losses first, long-term over short-term, wash-sale aware.",
    )
    target_defaults = {"Stocks": 35.0, "ETFs": 15.0, "Mutual Funds": 25.0, "Cash": 20.0, "Fixed Income": 5.0}
    targets: dict[str, float] = {}
    cols = st.columns(len(target_defaults))
    for i, asset in enumerate(target_defaults):
        with cols[i]:
            targets[asset] = st.number_input(f"{asset} target %", 0.0, 100.0, target_defaults[asset], 1.0) / 100
    # Share targets with the Action Center (read on the next rerun).
    st.session_state["reb_targets"] = targets
    total_target = sum(targets.values())
    if abs(total_target - 1) > 0.01:
        st.warning(f"Targets currently add to {total_target:.1%}. Make them total about 100%.")

    reb = allocation.copy()
    reb["Current Weight"] = reb["Market Value"] / broker_total if broker_total else np.nan
    reb["Target Weight"] = reb["Asset Class"].map(targets).fillna(0)
    reb["Target Value"] = reb["Target Weight"] * broker_total
    reb["Buy / Sell $"] = reb["Target Value"] - reb["Market Value"]
    with st.expander("Asset-class drift (dollar view)", expanded=False):
        st.dataframe(reb, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### 🎯 Trade generator")
    rebuy_raw = st.text_input(
        "Symbols you plan to rebuy (extends the wash-sale guard, comma-separated)", ""
    )
    rebuy_syms = tuple(s.strip().upper() for s in rebuy_raw.split(",") if s.strip())

    if st.button("Generate tax-aware plan", type="primary", use_container_width=True):
        if lots.empty:
            empty_state("No tax lots found — upload a positions file with lot-level detail.")
        else:
            plan = generate_plan(
                allocation, lots, summary, targets, broker_total,
                band_tolerance=reb_band, st_rate=st_tax_rate, lt_rate=lt_tax_rate,
                avoid_wash_sales=avoid_wash, rebuy_symbols=rebuy_syms,
            )
            st.session_state["reb_plan"] = plan

    plan = st.session_state.get("reb_plan")
    if plan is not None:
        s = plan.stats
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Net cash freed", fmt_currency(s["net_cash_freed"]))
        m2.metric("Realized gain / loss", fmt_currency(s["realized_total"]))
        m3.metric("Est. tax", fmt_currency(s["est_tax"]))
        m4.metric("Tax saved vs gains-only", fmt_currency(s["tax_saved_vs_gains_only"]))

        if plan.trades.empty:
            st.success("Already within your target bands — no trades needed.")
        else:
            st.markdown("##### Proposed blotter")
            blot = plan.trades.copy()
            for c in ["Est. $", "Realized G/L $", "Est. Tax $"]:
                blot[c] = blot[c].apply(lambda v: fmt_currency(v) if pd.notna(v) else "")
            st.dataframe(blot, use_container_width=True, hide_index=True)

            st.markdown("##### Resulting allocation")
            post = plan.post_allocation.copy()
            for c in ["Before Weight", "After Weight", "Target Weight"]:
                post[c] = post[c].apply(lambda v: fmt_percent(v, 1))
            post["In Band After"] = post["In Band After"].map({True: "✅", False: "⚠️"})
            st.dataframe(post, use_container_width=True, hide_index=True)

            for w in plan.warnings:
                st.warning(w)

            cdl, cai = st.columns(2)
            cdl.download_button(
                "📥 Blotter CSV", df_to_csv_bytes(plan.trades),
                "rebalance_blotter.csv", "text/csv", use_container_width=True,
            )
            if cai.button("🤖 Explain this plan", use_container_width=True):
                with st.spinner("Writing rationale…"):
                    st.markdown(ai_analyst.rebalance_rationale(
                        plan.trades, plan.stats, model=ai_model))

        st.caption(
            "⚠️ Estimates only — not tax advice. Wash-sale checks cover the 30-day "
            "window from your lot dates and rebuy list; verify substantially-identical "
            "rules and ST/LT netting before trading."
        )


# -----------------------------------------------------------------------------
# Tab: Smart Money (NEW INTEGRATION)
# -----------------------------------------------------------------------------
with tab_sm:
    section_header(
        "Smart Money / M-Block analyzer",
        "Auto-runs on your top holdings, then opens a manual analyzer for any ticker.",
    )

    sm_summary = auto_smart_money_table(summary, top_n=10)
    if not sm_summary.empty:
        st.markdown("#### Auto-read of your top 10 holdings")
        st.dataframe(
            style_cells(sm_summary, trend_cols=["Trend Rating"]),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Reads are derived from the EMA structure + drawdown context already on your "
            "holdings table. For full chart-level confirmation use the analyzer below."
        )
        st.divider()

    default_ticker = summary["Symbol"].iloc[0] if not summary.empty else "AAPL"
    render_smart_money_tab(default_ticker=default_ticker)


# -----------------------------------------------------------------------------
# Tab: Tax-Loss Harvesting (NEW)
# -----------------------------------------------------------------------------
with tab_tlh:
    section_header(
        "Tax-loss harvesting candidates",
        f"Lots with losses ≥ ${tlh_min_dollars:,.0f} or ≤ {tlh_min_pct:+.0%} (adjust thresholds in the sidebar).",
    )
    candidates = find_tlh_candidates(lots, min_loss_dollars=tlh_min_dollars, min_loss_pct=tlh_min_pct)
    if candidates.empty:
        empty_state("No tax lots found — upload a positions file with lot-level detail.")
    else:
        stats = tlh_summary(candidates)
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Candidates flagged", f"{stats['count']}")
        t2.metric("Total harvestable loss", fmt_currency(stats["total_loss"]))
        t3.metric("Long-term losses", fmt_currency(stats["lt_loss"]))
        t4.metric("Short-term losses", fmt_currency(stats["st_loss"]))

        st.dataframe(candidates, use_container_width=True, hide_index=True)
        st.caption(
            "⚠️ Wash-sale rule: the IRS disallows the loss if you buy a substantially identical "
            "security within 30 days before OR after the sale. This screen flags the loss only — "
            "you (or your advisor) are responsible for executing it correctly."
        )


# -----------------------------------------------------------------------------
# Tab: News & Earnings (NEW)
# -----------------------------------------------------------------------------
with tab_news:
    section_header("Headlines + upcoming earnings for your holdings")
    if not earnings.empty:
        st.markdown("#### Upcoming earnings")
        # Filter to next 60 days
        try:
            today = pd.Timestamp.today().normalize()
            upcoming = earnings[pd.to_datetime(earnings["Earnings Date"], errors="coerce")
                              .between(today, today + pd.Timedelta(days=60))]
            if not upcoming.empty:
                st.dataframe(upcoming, use_container_width=True, hide_index=True)
            else:
                st.caption("No earnings dates in the next 60 days.")
        except Exception:
            st.dataframe(earnings, use_container_width=True, hide_index=True)
    else:
        st.caption("Earnings calendar not available right now (yfinance not returning data).")

    st.markdown("#### Latest headlines per top holding")
    top_news_symbols = summary.head(8)["Symbol"].tolist()
    selected_sym = st.selectbox("Select a holding to see headlines", top_news_symbols)
    if selected_sym:
        items = cached_news(selected_sym)
        if not items:
            st.caption(f"No recent headlines available for {selected_sym}.")
        else:
            for item in items:
                title = item.get("title", "Untitled")
                publisher = item.get("publisher", "")
                link = item.get("link", "")
                summary_txt = item.get("summary", "")
                st.markdown(
                    f"""
                    <div class="epic-card" style="padding:12px 14px;margin-bottom:8px">
                        <a href="{link}" target="_blank" style="color:{PALETTE['text']};text-decoration:none;font-weight:600">
                            {title}
                        </a>
                        <div style="color:{PALETTE['text_muted']};font-size:0.78rem;margin-top:4px">
                            {publisher}
                        </div>
                        <div style="color:{PALETTE['text_muted']};font-size:0.85rem;margin-top:6px">
                            {summary_txt}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# -----------------------------------------------------------------------------
# Tab: Exports
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Tab: Performance (NEW) — attribution + backtest
# -----------------------------------------------------------------------------
with tab_perf:
    section_header("Performance attribution + backtest",
                   "Which holdings actually drove returns, and how you'd have done historically.")
    if not enable_advanced:
        empty_state("Turn on Advanced analytics in the sidebar.")
    elif returns.empty:
        empty_state("Turn on market data to compute performance metrics.")
    else:
        w_for_perf = summary.set_index("Symbol")["Portfolio Weight"].reindex(returns.columns).fillna(0)

        st.markdown("### Contribution to return (1Y)")
        attribution = performance_attribution(returns[[c for c in returns.columns if c != benchmark]], w_for_perf, period_days=252)
        if not attribution.empty:
            top_contrib = attribution.head(10)
            bot_contrib = attribution.tail(10).iloc[::-1]
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Top 10 contributors**")
                fig_top = px.bar(top_contrib, x="Contribution", y="Symbol", orientation="h",
                                 color="Contribution", color_continuous_scale="Greens",
                                 hover_data=["Weight", "Period Return"])
                fig_top.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=10),
                                     template="plotly_dark", yaxis_autorange="reversed",
                                     showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig_top, use_container_width=True)
            with col_b:
                st.markdown("**Top 10 detractors**")
                fig_bot = px.bar(bot_contrib, x="Contribution", y="Symbol", orientation="h",
                                 color="Contribution", color_continuous_scale="Reds_r",
                                 hover_data=["Weight", "Period Return"])
                fig_bot.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=10),
                                     template="plotly_dark", yaxis_autorange="reversed",
                                     showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig_bot, use_container_width=True)
            st.dataframe(attribution.head(30), use_container_width=True, hide_index=True)

        st.markdown("### Historical backtest")
        st.caption(f"Holds your CURRENT weights from {bt_start_date} to today. Rebalance: **{bt_rebalance}**.")
        bt_weights = summary.set_index("Symbol")["Portfolio Weight"]
        bt_weights = bt_weights[bt_weights > 0]
        if len(bt_weights) > 0:
            with st.spinner("Running backtest against historical data..."):
                bt_result = cached_backtest(
                    tuple(bt_weights.index.tolist()),
                    tuple(bt_weights.values.tolist()),
                    bt_start_date, None, benchmark, 100_000.0, bt_rebalance,
                )
            if bt_result:
                m = bt_result["metrics"]
                bt_c1, bt_c2, bt_c3, bt_c4 = st.columns(4)
                bt_c1.metric("Portfolio CAGR", fmt_percent(m["Portfolio CAGR"], 2),
                            fmt_percent(m["Excess CAGR"], 2, signed=True) + " vs benchmark")
                bt_c2.metric("Sharpe", f"{m['Portfolio Sharpe']:.2f}" if pd.notna(m['Portfolio Sharpe']) else "N/A",
                            f"vs {m['Benchmark Sharpe']:.2f}" if pd.notna(m['Benchmark Sharpe']) else None)
                bt_c3.metric("Max drawdown", fmt_percent(m["Portfolio Max DD"], 1))
                bt_c4.metric("Final value (from $100k)", fmt_currency(m["Final Portfolio Value"]),
                            fmt_currency(m["Final Portfolio Value"] - m["Final Benchmark Value"]) + " vs benchmark")

                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=bt_result["port_curve"].index, y=bt_result["port_curve"].values,
                                           name="Portfolio", line=dict(width=2, color=PALETTE["primary"])))
                fig_bt.add_trace(go.Scatter(x=bt_result["bench_curve"].index, y=bt_result["bench_curve"].values,
                                           name=benchmark, line=dict(width=2, color=PALETTE["info"], dash="dot")))
                fig_bt.update_layout(height=440, margin=dict(l=0, r=0, t=20, b=10),
                                    yaxis_title="Value ($)", template="plotly_dark",
                                    legend_orientation="h", hovermode="x unified")
                st.plotly_chart(fig_bt, use_container_width=True)

                with st.expander("Full backtest metrics"):
                    st.dataframe(pd.DataFrame([m]).T.rename(columns={0: "Value"}),
                                use_container_width=True)


# -----------------------------------------------------------------------------
# Tab: Projections (NEW) — Monte Carlo + Stress Test
# -----------------------------------------------------------------------------
with tab_proj:
    section_header("Forward projections + historical stress tests",
                   "What could happen ahead, and what HAS happened in past crises with your current weights.")
    if not enable_advanced:
        empty_state("Turn on Advanced analytics in the sidebar.")
    elif returns.empty:
        empty_state("Turn on market data to run projections.")
    else:
        w_for_mc = summary.set_index("Symbol")["Portfolio Weight"].reindex(returns.columns).fillna(0)
        port_only_returns = returns[[c for c in returns.columns if c != benchmark]]
        w_for_mc_clean = w_for_mc.reindex(port_only_returns.columns).fillna(0)

        st.markdown(f"### Monte Carlo projection — {mc_years} years, 5000 simulations")
        with st.spinner("Running 5000 bootstrap simulations..."):
            mc = monte_carlo_projection(
                port_only_returns, w_for_mc_clean, broker_total,
                years=mc_years, n_sims=5000, annual_contribution=mc_annual_contrib,
            )

        if mc is None:
            st.warning("Not enough return history to run Monte Carlo (need at least 60 days).")
        else:
            s = mc.summary
            mc_c1, mc_c2, mc_c3, mc_c4 = st.columns(4)
            mc_c1.metric("Median outcome", fmt_currency(s["final_p50"]),
                        fmt_percent(s["final_p50"]/broker_total - 1, 1, signed=True))
            mc_c2.metric("10th percentile", fmt_currency(s["final_p10"]),
                        fmt_percent(s["final_p10"]/broker_total - 1, 1, signed=True))
            mc_c3.metric("90th percentile", fmt_currency(s["final_p90"]),
                        fmt_percent(s["final_p90"]/broker_total - 1, 1, signed=True))
            mc_c4.metric("Prob. of doubling", fmt_percent(s["prob_double"], 1))

            mc_c5, mc_c6, mc_c7, mc_c8 = st.columns(4)
            mc_c5.metric("Prob. of tripling", fmt_percent(s["prob_triple"], 1))
            mc_c6.metric("Prob. of loss", fmt_percent(s["prob_loss"], 1))
            mc_c7.metric("Mean outcome", fmt_currency(s["final_mean"]))
            mc_c8.metric("25th–75th range",
                        f"{fmt_currency(s['final_p25'], 0)} to {fmt_currency(s['final_p75'], 0)}")

            x_days = np.arange(len(mc.percentile_50))
            x_years = x_days / 252
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(x=x_years, y=mc.percentile_90, name="90th percentile",
                                       line=dict(width=1, color=PALETTE["bullish"], dash="dot")))
            fig_mc.add_trace(go.Scatter(x=x_years, y=mc.percentile_50, name="Median",
                                       line=dict(width=2.5, color=PALETTE["primary"]),
                                       fill="tonexty", fillcolor="rgba(127,119,221,0.10)"))
            fig_mc.add_trace(go.Scatter(x=x_years, y=mc.percentile_10, name="10th percentile",
                                       line=dict(width=1, color=PALETTE["bearish"], dash="dot"),
                                       fill="tonexty", fillcolor="rgba(127,119,221,0.10)"))
            fig_mc.update_layout(height=420, margin=dict(l=0, r=0, t=20, b=10),
                                xaxis_title="Years from now", yaxis_title="Portfolio value ($)",
                                template="plotly_dark", legend_orientation="h", hovermode="x unified")
            st.plotly_chart(fig_mc, use_container_width=True)

            with st.expander("Custom target probability"):
                target = st.number_input("What's the probability I finish above this target?",
                                        0, 100_000_000, int(broker_total * 2), 10_000)
                prob = probability_of_target(mc, target)
                st.metric(f"Probability of reaching {fmt_currency(target)}", fmt_percent(prob, 1))

        st.divider()
        st.markdown("### Historical stress tests")
        st.caption("Applies your CURRENT weights to past crisis windows.")
        with st.spinner("Pulling crisis-era prices..."):
            stress_weights = summary.set_index("Symbol")["Portfolio Weight"]
            stress_weights = stress_weights[stress_weights > 0]
            stress_df = cached_stress_test(
                tuple(stress_weights.index.tolist()),
                tuple(stress_weights.values.tolist()),
                benchmark,
            )

        if stress_df.empty:
            st.caption("Stress data not available (yfinance may be rate-limited).")
        else:
            st.dataframe(stress_df, use_container_width=True, hide_index=True)
            fig_stress = px.bar(stress_df, x="Crisis", y=["Portfolio Return", f"{benchmark} Return"],
                               barmode="group",
                               color_discrete_map={"Portfolio Return": PALETTE["primary"],
                                                   f"{benchmark} Return": PALETTE["info"]})
            fig_stress.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=10),
                                    template="plotly_dark", yaxis_tickformat=".0%",
                                    legend_orientation="h")
            st.plotly_chart(fig_stress, use_container_width=True)


# -----------------------------------------------------------------------------
# Tab: Options (NEW) — covered-call screener + IV analysis
# -----------------------------------------------------------------------------
with tab_options:
    section_header("Options income screener",
                   "Find covered-call opportunities on stocks you already own.")
    if not enable_options:
        empty_state("Enable 'Options screener' in the sidebar (it's slower than other tabs).")
    else:
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            dte_min = st.slider("Min DTE", 7, 60, 25, 1)
        with oc2:
            dte_max = st.slider("Max DTE", 10, 90, 50, 1)
        with oc3:
            otm_target = st.slider("Target OTM %", 0, 25, 5, 1) / 100

        eligible = summary[
            (summary["Asset Type"] == "Stocks") & (summary["Shares"].fillna(0) >= 100)
        ][["Symbol", "Shares"]].dropna()

        if eligible.empty:
            empty_state("Need stock holdings of ≥100 shares to write covered calls.")
        else:
            st.caption(f"Screening {len(eligible)} stock positions with ≥100 shares each.")
            with st.spinner("Pulling options chains for all eligible holdings..."):
                cc_df = cached_covered_calls(
                    tuple(eligible["Symbol"].tolist()),
                    tuple(eligible["Shares"].tolist()),
                    dte_min, dte_max, otm_target,
                )
            if cc_df.empty:
                empty_state("No eligible options found in the selected DTE window.")
            else:
                total_premium = cc_df["Premium Income"].sum()
                avg_yield = cc_df["Annualized Premium Yield"].mean()
                ocm1, ocm2, ocm3 = st.columns(3)
                ocm1.metric("Eligible opportunities", str(len(cc_df)))
                ocm2.metric("Total premium if all written", fmt_currency(total_premium))
                ocm3.metric("Average annualized yield", fmt_percent(avg_yield, 1))

                # Display top opportunities
                display_df = cc_df.copy()
                display_df["Strike % OTM"] = display_df["Strike % OTM"].map(lambda x: f"{x:+.1%}")
                display_df["Annualized Premium Yield"] = display_df["Annualized Premium Yield"].map(lambda x: f"{x:.1%}")
                display_df["Implied Vol"] = display_df["Implied Vol"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                st.markdown("### IV vs realized volatility (top holdings)")
                with st.spinner("Computing IV/RV ratios..."):
                    iv_df = cached_iv_overview(tuple(eligible["Symbol"].head(10).tolist()))
                if not iv_df.empty:
                    st.session_state["iv_df"] = iv_df  # feed the Action Center
                    iv_display = iv_df.copy()
                    if "ATM IV" in iv_display: iv_display["ATM IV"] = iv_display["ATM IV"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
                    if "30D Realized Vol" in iv_display: iv_display["30D Realized Vol"] = iv_display["30D Realized Vol"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
                    if "IV/RV Ratio" in iv_display: iv_display["IV/RV Ratio"] = iv_display["IV/RV Ratio"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
                    st.dataframe(iv_display, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Tab: Factors (NEW) — factor exposure + efficient frontier
# -----------------------------------------------------------------------------
with tab_factors:
    section_header("Factor exposure + efficient frontier",
                   "Are you really diversified, or just buying growth in disguise?")
    if not enable_advanced:
        empty_state("Turn on Advanced analytics in the sidebar.")
    elif returns.empty:
        empty_state("Turn on market data to compute factor exposures.")
    else:
        st.markdown("### Style / factor exposure (OLS regression)")
        st.caption(
            "Regresses your daily portfolio returns against proxy ETFs for each factor. "
            "Betas > 0 = positive exposure; t-stats > 2 = statistically significant."
        )
        with st.spinner("Fetching factor proxy ETF returns..."):
            factor_returns = cached_factor_returns("2y")
        if factor_returns.empty:
            st.caption("Factor proxy data not available.")
        else:
            w_for_fx = summary.set_index("Symbol")["Portfolio Weight"].reindex(returns.columns).fillna(0)
            port_only = returns[[c for c in returns.columns if c != benchmark]]
            w_clean = w_for_fx.reindex(port_only.columns).fillna(0)
            if w_clean.sum() > 0:
                port_ret_series = port_only.mul(w_clean, axis=1).sum(axis=1)
                fx_result = factor_exposure(port_ret_series, factor_returns)
                if fx_result:
                    fc1, fc2 = st.columns([2, 1])
                    with fc1:
                        betas_df = pd.DataFrame([
                            {"Factor": k, "Beta": v,
                             "t-stat": fx_result["t_stats"].get(k, np.nan),
                             "Significant?": "✓" if abs(fx_result["t_stats"].get(k, 0)) > 2 else ""}
                            for k, v in fx_result["betas"].items()
                        ])
                        fig_fx = px.bar(betas_df, x="Factor", y="Beta", color="Beta",
                                       color_continuous_scale="RdBu", color_continuous_midpoint=0,
                                       text="Beta")
                        fig_fx.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                        fig_fx.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=10),
                                            template="plotly_dark", showlegend=False,
                                            coloraxis_showscale=False)
                        st.plotly_chart(fig_fx, use_container_width=True)
                    with fc2:
                        st.metric("Annualized alpha", fmt_percent(fx_result["alpha"], 2, signed=True))
                        st.metric("R-squared", f"{fx_result['r2']:.2%}")
                        st.metric("Observations", str(fx_result["n_obs"]))
                    st.dataframe(betas_df, use_container_width=True, hide_index=True)
                    # Cache for the AI analyst's "explain my factor exposures".
                    st.session_state["factor_table"] = betas_df
                    st.session_state["factor_alpha"] = fx_result["alpha"]
                    st.session_state["factor_r2"] = fx_result["r2"]
                    st.session_state["factor_nobs"] = fx_result["n_obs"]

        st.divider()
        st.markdown("### Efficient frontier (5,000 random portfolios)")
        st.caption(
            "Each dot is a random weight combination of your CURRENT holdings. "
            "Star = max Sharpe portfolio. Diamond = your current allocation."
        )
        port_only_for_ef = returns[[c for c in returns.columns if c != benchmark]]
        with st.spinner("Generating random-portfolio Monte Carlo..."):
            ef = efficient_frontier(port_only_for_ef, w_clean if w_clean.sum() > 0 else pd.Series(),
                                    n_portfolios=5000, risk_free=risk_free)
        if ef:
            fig_ef = go.Figure()
            fig_ef.add_trace(go.Scatter(
                x=ef["scatter_vols"] * 100, y=ef["scatter_returns"] * 100,
                mode="markers",
                marker=dict(size=4, color=ef["scatter_sharpes"], colorscale="Viridis",
                           showscale=True, colorbar=dict(title="Sharpe")),
                name="Random portfolios", opacity=0.6,
            ))
            fig_ef.add_trace(go.Scatter(
                x=[ef["max_sharpe"]["vol"] * 100], y=[ef["max_sharpe"]["return"] * 100],
                mode="markers", marker=dict(size=20, color=PALETTE["bullish"], symbol="star",
                                          line=dict(width=2, color="white")),
                name=f"Max Sharpe ({ef['max_sharpe']['sharpe']:.2f})",
            ))
            fig_ef.add_trace(go.Scatter(
                x=[ef["min_vol"]["vol"] * 100], y=[ef["min_vol"]["return"] * 100],
                mode="markers", marker=dict(size=16, color=PALETTE["info"], symbol="circle",
                                          line=dict(width=2, color="white")),
                name=f"Min Vol",
            ))
            if not np.isnan(ef["current"]["vol"]):
                fig_ef.add_trace(go.Scatter(
                    x=[ef["current"]["vol"] * 100], y=[ef["current"]["return"] * 100],
                    mode="markers", marker=dict(size=20, color=PALETTE["primary"], symbol="diamond",
                                              line=dict(width=2, color="white")),
                    name=f"Your portfolio ({ef['current']['sharpe']:.2f} Sharpe)",
                ))
            fig_ef.update_layout(
                height=520, margin=dict(l=0, r=0, t=30, b=10),
                xaxis_title="Annualized volatility (%)", yaxis_title="Annualized return (%)",
                template="plotly_dark", legend_orientation="h",
            )
            st.plotly_chart(fig_ef, use_container_width=True)

            with st.expander("Suggested weights — max-Sharpe portfolio"):
                ms_weights = pd.DataFrame([
                    {"Symbol": k, "Suggested Weight": v}
                    for k, v in sorted(ef["max_sharpe"]["weights"].items(), key=lambda x: -x[1])
                ])
                ms_weights["Suggested Weight"] = ms_weights["Suggested Weight"].map(lambda x: f"{x:.1%}")
                st.dataframe(ms_weights, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### Dividend growth analysis")
        income_symbols = summary[summary["Est. Annual Income"].fillna(0) > 0]["Symbol"].head(20).tolist()
        if income_symbols:
            with st.spinner("Pulling dividend histories..."):
                div_growth = cached_dividend_growth(tuple(income_symbols))
            if not div_growth.empty:
                div_display = div_growth.copy()
                if "5Y Dividend CAGR" in div_display:
                    div_display["5Y Dividend CAGR"] = div_display["5Y Dividend CAGR"].map(
                        lambda x: f"{x:+.1%}" if pd.notna(x) else "—")
                if "Growth Consistency" in div_display:
                    div_display["Growth Consistency"] = div_display["Growth Consistency"].map(
                        lambda x: f"{x:.0%}" if pd.notna(x) else "—")
                if "TTM Dividend" in div_display:
                    div_display["TTM Dividend"] = div_display["TTM Dividend"].map(
                        lambda x: f"${x:.2f}" if pd.notna(x) else "")
                st.dataframe(div_display, use_container_width=True, hide_index=True)
        else:
            st.caption("No income-paying holdings to analyze.")


# -----------------------------------------------------------------------------
# Tab: Macro (NEW) — market regime + macro indicators + insider/short data
# -----------------------------------------------------------------------------
with tab_macro:
    section_header("Market regime + macro indicators + insider/short data",
                   "Big picture context for deciding how aggressive or defensive to be.")

    if not enable_macro:
        empty_state("Enable 'macro + insider/short data' in the sidebar.")
    else:
        # ---- Market regime ----
        with st.spinner("Reading market regime signals..."):
            regime = cached_regime()
        if regime:
            regime_color = {"bullish": PALETTE["bullish"], "bearish": PALETTE["bearish"],
                           "warn": PALETTE["warning"], "neutral": PALETTE["neutral"]}.get(
                regime["color"], PALETTE["text_muted"])
            st.markdown(
                f"""
                <div class="epic-card" style="display:flex;align-items:center;gap:24px">
                    <div style="text-align:center;min-width:200px">
                        <div style="font-size:0.78rem;color:{PALETTE['text_muted']};letter-spacing:0.05em">MARKET REGIME</div>
                        <div style="font-size:1.8rem;font-weight:700;color:{regime_color};line-height:1.1;margin-top:8px">{regime['regime']}</div>
                        <div style="color:{PALETTE['text_muted']};font-size:0.78rem;margin-top:6px">
                            Risk-on signals: <b style="color:{PALETTE['bullish']}">{regime['score_on']}</b> /
                            Risk-off signals: <b style="color:{PALETTE['bearish']}">{regime['score_off']}</b>
                        </div>
                    </div>
                    <div style="flex:1">
                        <div style="font-size:0.85rem;color:{PALETTE['text_muted']};margin-bottom:6px">Signals fired:</div>
                        {"".join([f'<div style="margin-bottom:4px;font-size:0.88rem;color:{PALETTE["text"]}">• {n}</div>' for n in regime['notes']])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---- Macro indicators ----
        st.markdown("### Macro indicator dashboard")
        with st.spinner("Pulling VIX, yields, gold, credit spreads..."):
            macro = cached_macro()
        if not macro.empty:
            display_macro = macro.copy()
            for c in ["1D Change", "1W Change", "1M Change", "YTD Change"]:
                if c in display_macro.columns:
                    display_macro[c] = display_macro[c].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "")
            for c in ["Current", "6M High", "6M Low"]:
                if c in display_macro.columns:
                    display_macro[c] = display_macro[c].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
            st.dataframe(display_macro, use_container_width=True, hide_index=True)

            # Yield curve callout
            try:
                tnx = float(macro.loc[macro["Indicator"] == "10Y Treasury", "Current"].iloc[0])
                irx = float(macro.loc[macro["Indicator"] == "2Y Treasury", "Current"].iloc[0])
                spread = tnx - irx
                yc_color = PALETTE["bearish"] if spread < 0 else (PALETTE["warning"] if spread < 0.5 else PALETTE["bullish"])
                st.markdown(
                    f"<div style='color:{yc_color};font-weight:600'>"
                    f"Yield curve (10Y - 2Y): {spread:+.2f}pp "
                    f"{'(INVERTED — recession indicator)' if spread < 0 else '(flat — caution)' if spread < 0.5 else '(normal)'}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            except Exception:
                pass
        else:
            st.caption("Macro data not available.")

        st.divider()

        # ---- Insider activity ----
        st.markdown("### Insider activity (last 180 days)")
        st.caption("Net insider buying or selling across your stock holdings — when insiders buy with cash, it's a strong vote of confidence.")
        stock_syms = summary[
            (summary["Asset Type"] == "Stocks")
            & (~summary.get("Is Bond", pd.Series(False, index=summary.index)).fillna(False).astype(bool))
        ]["Symbol"].head(20).tolist()

        if stock_syms:
            with st.spinner("Pulling insider transactions..."):
                ins_df = cached_insider(tuple(stock_syms))
            if not ins_df.empty:
                # Sort to surface largest net buying first
                ins_display = ins_df.sort_values("Net Insider Activity $", ascending=False).copy()
                for c in ["Buy Value", "Sell Value", "Net Insider Activity $"]:
                    if c in ins_display.columns:
                        ins_display[c] = ins_display[c].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
                st.dataframe(ins_display, use_container_width=True, hide_index=True)
            else:
                st.caption("No insider activity data available.")

        # ---- Short interest ----
        st.markdown("### Short interest snapshot")
        st.caption("High short interest can mean either a brewing short squeeze or that smart money is betting against the stock.")
        if stock_syms:
            with st.spinner("Pulling short-interest data..."):
                si_df = cached_short_interest(tuple(stock_syms))
            if not si_df.empty:
                si_display = si_df.sort_values("Short % of Float", ascending=False, na_position="last").copy()
                if "Short % of Float" in si_display.columns:
                    si_display["Short % of Float"] = si_display["Short % of Float"].map(
                        lambda x: f"{x:.1%}" if pd.notna(x) else "")
                st.dataframe(si_display, use_container_width=True, hide_index=True)
            else:
                st.caption("Short interest data not available.")

        # ---- Analyst rating changes ----
        st.markdown("### Recent analyst rating changes")
        if stock_syms:
            with st.spinner("Pulling analyst upgrades/downgrades..."):
                ac_df = cached_analyst_changes(tuple(stock_syms[:10]))
            if not ac_df.empty:
                st.dataframe(ac_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No recent analyst changes found.")


# -----------------------------------------------------------------------------
# Tab: Themes (NEW) — correlation-aware clusters + rebalance simulator
# -----------------------------------------------------------------------------
with tab_themes:
    section_header(
        "Theme clusters + tax-aware rebalance simulator",
        "Your sector view splits correlated bets apart. This groups them by THEME so you see your true concentration.",
    )

    themed = assign_themes(summary)
    te = theme_exposure(themed, broker_total)

    if te.empty:
        empty_state("No holdings to cluster.")
    else:
        # --- Theme exposure chart + HHI ---
        hhi_info = theme_concentration_hhi(te)
        tc1, tc2 = st.columns([2, 1])
        with tc1:
            fig_theme = px.treemap(
                te, path=["Theme"], values="Market Value",
                color="Weight", color_continuous_scale="Sunsetdark",
                title="True exposure by theme (correlated clusters)",
            )
            fig_theme.update_traces(texttemplate="<b>%{label}</b><br>%{value:$,.0f}")
            fig_theme.update_layout(height=440, margin=dict(l=0, r=0, t=40, b=10),
                                   template="plotly_dark")
            st.plotly_chart(fig_theme, use_container_width=True)
        with tc2:
            if hhi_info:
                st.metric("Theme HHI (risk assets)", f"{hhi_info['hhi']:.3f}",
                         help="Concentration across themes, excluding cash/bonds. "
                              "Lower is more diversified. >0.25 = concentrated.")
                st.metric("Effective # of themes", f"{hhi_info['effective_themes']:.1f}")
                st.metric("Largest theme",
                         f"{hhi_info['largest_theme_weight']:.1%}",
                         help=hhi_info['largest_theme'])
                # Compare to the naive sector HHI for contrast
                _sector = sector_concentration_hhi(
                    summary[summary["Asset Type"].isin(["Stocks", "ETFs"])]
                ) if "Sector" in summary.columns else pd.DataFrame()
                if not _sector.empty:
                    sec_hhi = float((_sector["Portfolio Weight"] ** 2).sum())
                    st.caption(
                        f"Your sector HHI looks like **{sec_hhi:.3f}** "
                        f"(~{(1/sec_hhi if sec_hhi else 0):.0f} sectors), but theme HHI is "
                        f"**{hhi_info['hhi']:.3f}** (~{hhi_info['effective_themes']:.0f} real bets). "
                        "The gap is your hidden concentration."
                    )

        # --- Theme detail table ---
        te_disp = te.copy()
        te_disp["Market Value"] = te_disp["Market Value"].map(lambda x: f"${x:,.0f}")
        te_disp["Weight"] = te_disp["Weight"].map(lambda x: f"{x:.1%}")
        te_disp["Unrealized P&L $"] = te_disp["Unrealized P&L $"].map(lambda x: f"${x:+,.0f}")
        st.dataframe(te_disp, use_container_width=True, hide_index=True)

        # --- Per-theme holdings drilldown ---
        with st.expander("See holdings inside each theme"):
            for theme in te["Theme"]:
                members = themed[themed["Theme"] == theme][
                    ["Symbol", "Description", "Market Value", "Portfolio Weight", "Unrealized P&L $"]
                ].sort_values("Market Value", ascending=False)
                if not members.empty:
                    st.markdown(f"**{theme}** — {len(members)} holding(s)")
                    st.dataframe(members, use_container_width=True, hide_index=True)

        st.divider()

        # =====================================================================
        # Rebalance simulator
        # =====================================================================
        section_header(
            "Tax-aware rebalance simulator",
            "Model trimming winners + harvesting losers. See the realized gain/loss, tax bill, and resulting theme exposure BEFORE you trade.",
        )

        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            trim_threshold = st.slider("Trim positions up more than (%)", 10, 200, 50, 5) / 100
        with sc2:
            trim_amount = st.slider("Trim this % of each winner", 10, 100, 30, 5) / 100
        with sc3:
            harvest_min = st.number_input("Harvest losses beyond ($)", 0, 10000, 250, 50)
        with sc4:
            lt_rate = st.slider("Your LT cap-gains rate (%)", 0, 40, 15, 1) / 100

        st_rate = st.slider("Your short-term / ordinary rate (%)", 0, 50, 24, 1) / 100

        # Make sure Total Cost exists for the suggester
        themed_for_sim = themed.copy()
        if "Total Cost" not in themed_for_sim.columns:
            themed_for_sim["Total Cost"] = themed_for_sim.get("Total Cost", np.nan)

        auto_actions = suggest_rebalance_actions(
            themed_for_sim,
            trim_gain_threshold=trim_threshold,
            trim_pct=trim_amount,
            harvest_loss_dollars=harvest_min,
        )

        if not auto_actions:
            st.info("No positions currently meet the trim/harvest thresholds. Loosen the sliders to see suggestions.")
        else:
            st.markdown(f"#### Suggested actions ({len(auto_actions)})")
            st.caption("These are auto-generated from your thresholds. The simulation below shows the combined effect.")

            sim = simulate_rebalance(
                themed_for_sim, auto_actions,
                total_portfolio=broker_total,
                lt_cap_gains_rate=lt_rate, st_cap_gains_rate=st_rate,
            )

            # Headline impact metrics
            rm1, rm2, rm3, rm4 = st.columns(4)
            rm1.metric("Cash freed", fmt_currency(sim.total_proceeds))
            rm2.metric("Net realized P&L", fmt_currency(sim.net_realized),
                      "gain" if sim.net_realized >= 0 else "loss")
            rm3.metric("Estimated tax", fmt_currency(sim.estimated_tax))
            carryf = getattr(sim, "_carryforward_loss", 0.0)
            rm4.metric("Loss carryforward", fmt_currency(carryf) if carryf else "$0")

            # Actions table
            actions_df = pd.DataFrame(sim.actions)
            if not actions_df.empty:
                disp = actions_df.copy()
                disp["% Sold"] = disp["% Sold"].map(lambda x: f"{x:.0%}")
                disp["Proceeds"] = disp["Proceeds"].map(lambda x: f"${x:,.0f}")
                disp["Realized P&L"] = disp["Realized P&L"].map(lambda x: f"${x:+,.0f}")
                st.dataframe(disp, use_container_width=True, hide_index=True)

            # Before/after theme exposure
            st.markdown("#### Theme exposure: before vs after")
            if sim.theme_before is not None and sim.theme_after is not None:
                before = sim.theme_before[["Theme", "Weight"]].rename(columns={"Weight": "Before"})
                after = sim.theme_after[["Theme", "Weight"]].rename(columns={"Weight": "After"})
                merged = before.merge(after, on="Theme", how="outer").fillna(0)
                merged["Change"] = merged["After"] - merged["Before"]
                merged = merged.sort_values("Before", ascending=False)

                fig_ba = go.Figure()
                fig_ba.add_trace(go.Bar(
                    y=merged["Theme"], x=merged["Before"] * 100, name="Before",
                    orientation="h", marker_color=PALETTE["neutral"],
                ))
                fig_ba.add_trace(go.Bar(
                    y=merged["Theme"], x=merged["After"] * 100, name="After",
                    orientation="h", marker_color=PALETTE["primary"],
                ))
                fig_ba.update_layout(
                    height=420, margin=dict(l=0, r=0, t=20, b=10),
                    barmode="group", template="plotly_dark",
                    xaxis_title="Portfolio weight (%)", legend_orientation="h",
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_ba, use_container_width=True)

                merged_disp = merged.copy()
                for c in ["Before", "After", "Change"]:
                    merged_disp[c] = merged_disp[c].map(lambda x: f"{x:+.1%}" if c == "Change" else f"{x:.1%}")
                st.dataframe(merged_disp, use_container_width=True, hide_index=True)

            st.caption(
                "⚠️ Realized P&L is estimated pro-rata across each position (not lot-specific). "
                "Actual tax depends on which tax lots you sell and the wash-sale rule (can't rebuy a "
                "substantially identical security within 30 days of a harvested loss). This is a planning "
                "tool, not tax advice — confirm with your advisor before trading."
            )


# -----------------------------------------------------------------------------
# Tab: AI Analyst  (NEW — Claude-powered narrative layer)
# -----------------------------------------------------------------------------
with tab_ai:
    section_header(
        "AI analyst",
        "Claude turns your numbers into narrative: portfolio briefings, plain-English "
        "factor readouts, and per-name theses. Educational analysis, not advice.",
    )
    _ready, _msg = ai_analyst.availability()
    if not _ready:
        st.info(_msg)
        st.markdown(
            "**To enable:** add `anthropic` to `requirements.txt` and set "
            "`ANTHROPIC_API_KEY` in **Settings → Secrets** on Streamlit Cloud. "
            "Your key stays in your runtime and is never logged."
        )
    else:
        st.caption(f"Using **{ai_model_label}** — change the model in the sidebar.")

        c1, c2 = st.columns(2)
        if c1.button("📋 Portfolio briefing", use_container_width=True):
            with st.spinner("Analyzing the book…"):
                st.session_state["ai_brief"] = ai_analyst.portfolio_commentary(
                    summary, allocation, broker_total=broker_total,
                    ugl_total=ugl_total, hhi=_hhi, model=ai_model,
                )
        if c2.button("🔬 Explain my factor exposures", use_container_width=True):
            ft = st.session_state.get("factor_table")
            if ft is None:
                st.warning("Open the Factors tab once this session to compute exposures first.")
            else:
                with st.spinner("Reading the regression…"):
                    st.session_state["ai_brief"] = ai_analyst.explain_factors(
                        ft, alpha=st.session_state.get("factor_alpha"),
                        r2=st.session_state.get("factor_r2"),
                        n_obs=st.session_state.get("factor_nobs"), model=ai_model,
                    )
        if st.session_state.get("ai_brief"):
            st.markdown(st.session_state["ai_brief"])

        st.divider()
        st.markdown("#### Single-name thesis")
        syms = summary["Symbol"].dropna().tolist()
        pick = st.selectbox("Holding", syms) if syms else None
        if pick and st.button(f"Generate {pick} thesis", use_container_width=True):
            row = summary[summary["Symbol"].eq(pick)].iloc[0].to_dict()
            with st.spinner(f"Writing {pick} thesis…"):
                st.markdown(ai_analyst.thesis(pick, row, model=ai_model))


# -----------------------------------------------------------------------------
# Tab: Exports
# -----------------------------------------------------------------------------
with tab_exp:
    section_header("Download cleaned + enriched files")

    summary_csv = df_to_csv_bytes(summary)
    lots_csv = df_to_csv_bytes(lots) if not lots.empty else b""
    alloc_csv = df_to_csv_bytes(allocation)
    tv_consolidated = df_to_csv_bytes(tradingview_csv(result.summary, "consolidated"))
    tv_lot = df_to_csv_bytes(tradingview_csv(lots, "lot")) if not lots.empty else b""

    tlh_df = find_tlh_candidates(lots, min_loss_dollars=tlh_min_dollars, min_loss_pct=tlh_min_pct)
    tlh_csv = df_to_csv_bytes(tlh_df) if not tlh_df.empty else b""

    d1, d2, d3 = st.columns(3)
    d1.download_button("📥 Enriched holdings CSV", summary_csv, "portfolio_holdings_enriched.csv", "text/csv", use_container_width=True)
    if lots_csv:
        d2.download_button("📥 Tax lots CSV", lots_csv, "portfolio_lot_level_cleaned.csv", "text/csv", use_container_width=True)
    d3.download_button("📥 Allocation CSV", alloc_csv, "portfolio_allocation.csv", "text/csv", use_container_width=True)

    d4, d5, d6 = st.columns(3)
    d4.download_button("📥 TradingView (consolidated)", tv_consolidated, "tradingview_portfolio_import_consolidated.csv", "text/csv", use_container_width=True)
    if tv_lot:
        d5.download_button("📥 TradingView (lot)", tv_lot, "tradingview_portfolio_import_lot_level.csv", "text/csv", use_container_width=True)
    if tlh_csv:
        d6.download_button("📥 TLH candidates", tlh_csv, "tax_loss_harvesting_candidates.csv", "text/csv", use_container_width=True)

    # ZIP bundle
    zip_files = {
        "portfolio_holdings_enriched.csv": summary_csv,
        "portfolio_lot_level_cleaned.csv": lots_csv,
        "portfolio_allocation.csv": alloc_csv,
        "tradingview_portfolio_import_consolidated.csv": tv_consolidated,
        "tradingview_portfolio_import_lot_level.csv": tv_lot,
        "tax_loss_harvesting_candidates.csv": tlh_csv,
    }
    zip_bytes = build_zip(zip_files)

    bundle_col, pdf_col = st.columns(2)
    bundle_col.download_button(
        "📦 Download all CSVs as ZIP",
        zip_bytes, "portfolio_epic_outputs.zip", "application/zip",
        use_container_width=True,
    )

    # PDF report
    if pdf_available():
        sector_df_for_pdf = sector_concentration_hhi(summary) if use_market_data else None
        pdf_bytes = build_pdf_report(
            broker_total=broker_total, ugl_total=ugl_total,
            today_total=today_total, income_total=income_total,
            cash_weight=cash_weight, top1_weight=top1_weight, top5_weight=top5_weight,
            flag_items=flag_items, allocation=allocation,
            top_holdings=summary, sector_breakdown=sector_df_for_pdf,
            tlh_candidates=tlh_df,
        )
        if pdf_bytes:
            pdf_col.download_button(
                "📄 Download dashboard summary as PDF",
                pdf_bytes, "portfolio_epic_summary.pdf", "application/pdf",
                use_container_width=True,
            )
    else:
        pdf_col.caption("Install `fpdf2` to enable PDF export: `pip install fpdf2`")
