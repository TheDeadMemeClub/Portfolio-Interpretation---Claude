"""Smart Money / M-Block trade planner.

Cleaned-up version of the original `smart_money_feature.py`. Same trading rules
(Structure > SND > Wyckoff, M-Blocks, POIs) and same scoring system, but:
  * Constants are module-level (no magic strings buried in render code)
  * `render_smart_money_tab` accepts a `default_ticker` from the portfolio so
    it auto-loads the user's largest holding instead of always AAPL
  * Trade journal entries can be appended to `st.session_state` and exported
    in bulk, not just one trade at a time
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None  # type: ignore


PERIOD_OPTIONS: dict[str, str] = {
    "1D": "1d", "5D": "5d", "1M": "1mo", "3M": "3mo", "6M": "6mo",
    "YTD": "ytd", "1Y": "1y", "2Y": "2y", "5Y": "5y", "10Y": "10y", "MAX": "max",
}

INTERVAL_OPTIONS: dict[str, str] = {
    "1m": "1m", "2m": "2m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "60m", "4h": "4h", "1d": "1d", "1wk": "1wk", "1mo": "1mo",
}

# Trade grading weights (total = 100)
TRADE_WEIGHTS: dict[str, int] = {
    "HTF bias aligned": 20,
    "Structure confirms direction": 20,
    "Valid POI / SND zone": 15,
    "MB / DMB confirmation": 15,
    "Wyckoff confirmation": 10,
    "Clean 1:3+ risk/reward": 10,
    "No major news / low-liquidity issue": 5,
    "Partial + breakeven plan": 5,
}

JOURNAL_KEY = "smart_money_journal"


@dataclass
class TradeInputs:
    ticker: str
    direction: str
    account_size: float
    risk_pct: float
    entry: float
    stop: float
    target: float


# ---------------------------------------------------------------------------
# Data + indicators
# ---------------------------------------------------------------------------
def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")
    df = yf.download(
        ticker, period=period, interval=interval,
        auto_adjust=False, progress=False, threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = _flatten_yf_columns(df).reset_index()
    date_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={date_col: "Date"})
    for col in ("Date", "Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            df[col] = np.nan
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for span in (10, 20, 50, 200):
        out[f"EMA{span}"] = out["Close"].ewm(span=span, adjust=False).mean()
    out["Range"] = out["High"] - out["Low"]
    out["Body"] = (out["Close"] - out["Open"]).abs()
    out["BodyPctRange"] = np.where(out["Range"] > 0, out["Body"] / out["Range"], 0)
    out["AvgRange20"] = out["Range"].rolling(20, min_periods=5).mean()
    out["AvgVol20"] = out["Volume"].rolling(20, min_periods=5).mean()
    return out


def detect_swings(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    out = df.copy()
    out["SwingHigh"] = False
    out["SwingLow"] = False
    if len(out) < window * 2 + 1:
        return out
    highs, lows = out["High"].values, out["Low"].values
    for i in range(window, len(out) - window):
        out.loc[out.index[i], "SwingHigh"] = highs[i] == np.max(highs[i - window: i + window + 1])
        out.loc[out.index[i], "SwingLow"] = lows[i] == np.min(lows[i - window: i + window + 1])
    return out


def classify_trend(df: pd.DataFrame) -> str:
    if df.empty or len(df) < 60:
        return "Not enough data"
    last = df.iloc[-1]
    bullish_ema = last["EMA10"] > last["EMA20"] > last["EMA50"] and last["Close"] > last["EMA200"]
    bearish_ema = last["EMA10"] < last["EMA20"] < last["EMA50"] and last["Close"] < last["EMA200"]
    swing_highs = df[df["SwingHigh"]].tail(3)
    swing_lows = df[df["SwingLow"]].tail(3)
    higher_lows = len(swing_lows) >= 2 and swing_lows["Low"].iloc[-1] > swing_lows["Low"].iloc[-2]
    higher_highs = len(swing_highs) >= 2 and swing_highs["High"].iloc[-1] > swing_highs["High"].iloc[-2]
    lower_lows = len(swing_lows) >= 2 and swing_lows["Low"].iloc[-1] < swing_lows["Low"].iloc[-2]
    lower_highs = len(swing_highs) >= 2 and swing_highs["High"].iloc[-1] < swing_highs["High"].iloc[-2]
    if bullish_ema and (higher_lows or higher_highs):
        return "Bullish"
    if bearish_ema and (lower_lows or lower_highs):
        return "Bearish"
    if bullish_ema:
        return "Bullish EMA alignment, structure needs confirmation"
    if bearish_ema:
        return "Bearish EMA alignment, structure needs confirmation"
    return "Neutral / Choppy"


def detect_bos(df: pd.DataFrame) -> tuple[str, Optional[float]]:
    if df.empty or len(df) < 30:
        return "No clear BOS", None
    last_close = float(df["Close"].iloc[-1])
    prev_swing_highs = df[df["SwingHigh"]].iloc[:-1].tail(3)
    prev_swing_lows = df[df["SwingLow"]].iloc[:-1].tail(3)
    if not prev_swing_highs.empty:
        recent_high = float(prev_swing_highs["High"].max())
        if last_close > recent_high:
            return "Bullish BOS / displacement body close", recent_high
    if not prev_swing_lows.empty:
        recent_low = float(prev_swing_lows["Low"].min())
        if last_close < recent_low:
            return "Bearish BOS / displacement body close", recent_low
    return "No fresh body-close BOS", None


def detect_market_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Heuristic detector: displacement candles > 1.25x avg range, body > 55%."""
    out = df.copy()
    out["Displacement"] = (out["Range"] > 1.25 * out["AvgRange20"]) & (out["BodyPctRange"] > 0.55)
    out["BullishDisplacement"] = out["Displacement"] & (out["Close"] > out["Open"])
    out["BearishDisplacement"] = out["Displacement"] & (out["Close"] < out["Open"])
    out["PotentialDemandMB"] = False
    out["PotentialSupplyMB"] = False
    for i in range(2, len(out)):
        if out["BullishDisplacement"].iloc[i]:
            for j in range(i - 1, max(i - 6, -1), -1):
                if out["Close"].iloc[j] < out["Open"].iloc[j]:
                    out.loc[out.index[j], "PotentialDemandMB"] = True
                    break
        if out["BearishDisplacement"].iloc[i]:
            for j in range(i - 1, max(i - 6, -1), -1):
                if out["Close"].iloc[j] > out["Open"].iloc[j]:
                    out.loc[out.index[j], "PotentialSupplyMB"] = True
                    break
    return out


def recent_zones(df: pd.DataFrame, max_zones: int = 8) -> pd.DataFrame:
    zones: list[dict] = []
    for _, row in df[df["PotentialDemandMB"]].tail(max_zones).iterrows():
        zones.append({
            "Date": row["Date"], "Type": "Demand MB / possible SND",
            "Low": float(row["Low"]), "High": float(row["High"]),
            "50%": float((row["Low"] + row["High"]) / 2),
            "Status": "Manual confirm: valid until body close below zone",
        })
    for _, row in df[df["PotentialSupplyMB"]].tail(max_zones).iterrows():
        zones.append({
            "Date": row["Date"], "Type": "Supply MB / possible SND",
            "Low": float(row["Low"]), "High": float(row["High"]),
            "50%": float((row["Low"] + row["High"]) / 2),
            "Status": "Manual confirm: valid until body close above zone",
        })
    if not zones:
        return pd.DataFrame(columns=["Date", "Type", "Low", "High", "50%", "Status"])
    return pd.DataFrame(zones).sort_values("Date", ascending=False).head(max_zones)


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
def make_chart(df: pd.DataFrame, ticker: str, zones_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name=ticker.upper(),
    ))
    ema_colors = {"EMA10": "#7DD3FC", "EMA20": "#A78BFA", "EMA50": "#F59E0B", "EMA200": "#EF4444"}
    for ema, color in ema_colors.items():
        if ema in df.columns and df[ema].notna().any():
            fig.add_trace(go.Scatter(
                x=df["Date"], y=df[ema], name=ema, mode="lines",
                line=dict(width=1.2, color=color),
            ))
    highs, lows = df[df["SwingHigh"]], df[df["SwingLow"]]
    fig.add_trace(go.Scatter(
        x=highs["Date"], y=highs["High"], mode="markers", name="Swing High",
        marker=dict(symbol="triangle-down", size=8, color="#FCA5A5"),
    ))
    fig.add_trace(go.Scatter(
        x=lows["Date"], y=lows["Low"], mode="markers", name="Swing Low",
        marker=dict(symbol="triangle-up", size=8, color="#86EFAC"),
    ))
    if not zones_df.empty:
        for _, z in zones_df.head(5).iterrows():
            color = "rgba(34,197,94,0.10)" if "Demand" in z["Type"] else "rgba(239,68,68,0.10)"
            fig.add_hrect(
                y0=z["Low"], y1=z["High"], line_width=1, fillcolor=color, opacity=0.6,
                annotation_text=z["Type"], annotation_position="top left",
            )
    fig.update_layout(
        title=f"{ticker.upper()} — Smart Money / M-Block analysis",
        xaxis_title="Date", yaxis_title="Price", height=620,
        xaxis_rangeslider_visible=False, legend_orientation="h",
        margin=dict(l=10, r=10, t=50, b=10),
        template="plotly_dark",
    )
    return fig


# ---------------------------------------------------------------------------
# Risk math + grading
# ---------------------------------------------------------------------------
def calc_risk(inp: TradeInputs) -> dict[str, float | str]:
    entry, stop, target = inp.entry, inp.stop, inp.target
    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0 or inp.account_size <= 0 or inp.risk_pct <= 0:
        return {"error": "Entry, stop, account size, and risk must be valid."}
    dollar_risk = inp.account_size * (inp.risk_pct / 100)
    shares = float(np.floor(dollar_risk / risk_per_share))
    if inp.direction == "Long":
        reward_per_share = target - entry
        one_r, two_r, three_r = entry + risk_per_share, entry + 2 * risk_per_share, entry + 3 * risk_per_share
    else:
        reward_per_share = entry - target
        one_r, two_r, three_r = entry - risk_per_share, entry - 2 * risk_per_share, entry - 3 * risk_per_share
    rr = reward_per_share / risk_per_share if risk_per_share else np.nan
    return {
        "Dollar Risk": round(dollar_risk, 2),
        "Risk / Share": round(risk_per_share, 4),
        "Suggested Shares": int(max(shares, 0)),
        "Position Value": round(shares * entry, 2),
        "Potential Reward / Share": round(reward_per_share, 4),
        "R:R": round(rr, 2),
        "1R Level": round(one_r, 4),
        "2R Level": round(two_r, 4),
        "3R Level": round(three_r, 4),
    }


def score_trade(checks: dict[str, bool]) -> tuple[int, str]:
    score = sum(TRADE_WEIGHTS[k] for k, v in checks.items() if v)
    if score >= 90:
        grade = "A+"
    elif score >= 80:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 60:
        grade = "C / only if very clean"
    else:
        grade = "NO TRADE"
    return score, grade


def _journal_row(**kwargs) -> dict:
    row = {"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    row.update(kwargs)
    return row


# ---------------------------------------------------------------------------
# Streamlit render
# ---------------------------------------------------------------------------
def render_smart_money_tab(default_ticker: str = "AAPL") -> None:
    st.subheader("Smart Money / M-Block stock analyzer")
    st.caption(
        "Rule-based planner built around Structure > SND > Wyckoff, M-Blocks, POIs, "
        "risk management, partials, and backtesting discipline."
    )

    col_a, col_b, col_c = st.columns([1.2, 1, 1])
    with col_a:
        ticker = st.text_input("Ticker", value=default_ticker, key="sm_ticker").strip().upper()
    with col_b:
        period_label = st.selectbox(
            "Lookback", list(PERIOD_OPTIONS.keys()), index=6, key="sm_period"
        )
    with col_c:
        interval_label = st.selectbox(
            "Candle timeframe", list(INTERVAL_OPTIONS.keys()), index=7, key="sm_interval"
        )

    period = PERIOD_OPTIONS[period_label]
    interval = INTERVAL_OPTIONS[interval_label]

    if not ticker:
        st.warning("Enter a ticker to begin.")
        return

    try:
        raw = fetch_ohlcv(ticker, period, interval)
    except Exception as exc:
        st.error(f"Could not fetch chart data: {exc}")
        return
    if raw.empty:
        st.error("No data returned. Try a different ticker, lookback, or timeframe.")
        return

    df = add_indicators(raw)
    df = detect_swings(df, window=3)
    df = detect_market_blocks(df)

    bias = classify_trend(df)
    bos_text, _ = detect_bos(df)
    zones_df = recent_zones(df)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current price", f"${df['Close'].iloc[-1]:,.2f}")
    m2.metric("Bias", bias)
    m3.metric("Structure", bos_text)
    m4.metric("Detected zones", len(zones_df))

    st.plotly_chart(make_chart(df, ticker, zones_df), use_container_width=True)

    with st.expander("Recent possible MB / SND zones", expanded=True):
        if zones_df.empty:
            st.info("No recent zones detected — manually mark your POI from the chart.")
        else:
            st.dataframe(zones_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Trade-grade checklist")
    st.caption("The app should not force trades. It keeps you out unless the setup is clean.")

    c1, c2 = st.columns(2)
    with c1:
        htf = st.checkbox("HTF bias aligned", value=("Bullish" in bias or "Bearish" in bias), key="chk_htf")
        structure = st.checkbox("Structure confirms direction", value=("BOS" in bos_text), key="chk_struct")
        poi = st.checkbox("Valid POI / SND zone", key="chk_poi")
        mb = st.checkbox("MB / DMB confirmation", key="chk_mb")
    with c2:
        wyckoff = st.checkbox("Wyckoff confirmation", key="chk_wyck")
        rr_clean = st.checkbox("Clean 1:3+ risk/reward", key="chk_rr")
        no_news = st.checkbox("No major news / low-liquidity issue", value=True, key="chk_news")
        partial_plan = st.checkbox("Partial + breakeven plan", value=True, key="chk_part")

    checks = {
        "HTF bias aligned": htf, "Structure confirms direction": structure,
        "Valid POI / SND zone": poi, "MB / DMB confirmation": mb,
        "Wyckoff confirmation": wyckoff, "Clean 1:3+ risk/reward": rr_clean,
        "No major news / low-liquidity issue": no_news,
        "Partial + breakeven plan": partial_plan,
    }
    score, grade = score_trade(checks)
    g1, g2 = st.columns(2)
    g1.metric("Setup score", f"{score}/100")
    g2.metric("Setup grade", grade)
    if grade == "NO TRADE":
        st.error("NO TRADE: setup does not have enough confirmation under your rules.")
    elif score < 75:
        st.warning("Low-to-medium quality setup. Consider waiting for cleaner confirmation.")
    else:
        st.success("Higher-quality setup. Still confirm manually before trading.")

    st.divider()
    st.subheader("Risk + position-size calculator")

    r1, r2, r3, r4, r5 = st.columns(5)
    with r1:
        direction = st.selectbox("Direction", ["Long", "Short"], key="sm_dir")
    with r2:
        account_size = st.number_input("Account size", min_value=0.0, value=10000.0, step=500.0, key="sm_acct")
    with r3:
        risk_pct = st.number_input("Risk %", min_value=0.01, max_value=10.0, value=0.5, step=0.1, key="sm_risk")
    with r4:
        entry = st.number_input(
            "Entry", min_value=0.0, value=float(df["Close"].iloc[-1]),
            step=0.01, format="%.4f", key="sm_entry",
        )
    with r5:
        stop = st.number_input(
            "Stop", min_value=0.0, value=float(df["Low"].tail(20).min()),
            step=0.01, format="%.4f", key="sm_stop",
        )

    default_target = (entry + 3 * abs(entry - stop)) if direction == "Long" else max(entry - 3 * abs(entry - stop), 0.0)
    target = st.number_input("Target", min_value=0.0, value=float(default_target),
                             step=0.01, format="%.4f", key="sm_target")

    risk_data = calc_risk(TradeInputs(ticker, direction, account_size, risk_pct, entry, stop, target))
    if "error" in risk_data:
        st.error(str(risk_data["error"]))
    else:
        st.dataframe(pd.DataFrame([risk_data]), use_container_width=True, hide_index=True)
        if float(risk_data["R:R"]) >= 3:
            st.success("R:R passes the 1:3 target rule.")
        else:
            st.warning("R:R is below 1:3. Improve entry, tighten invalidation, or skip.")

    st.divider()
    st.subheader("Trade journal")

    notes = st.text_area(
        "Setup notes",
        placeholder=(
            "Example: HTF bullish, price retraced into demand MB, MB2 reconfirmed, "
            "possible reaccumulation, entry on confirmation, stop below MB low."
        ),
        height=100, key="sm_notes",
    )

    new_row = _journal_row(
        Ticker=ticker, Direction=direction, Timeframe=interval_label,
        Lookback=period_label, Bias=bias, BOS=bos_text,
        **{"Setup Score": score, "Setup Grade": grade, "Notes": notes},
        **(risk_data if "error" not in risk_data else {}),
    )

    if JOURNAL_KEY not in st.session_state:
        st.session_state[JOURNAL_KEY] = []

    j1, j2 = st.columns([1, 1])
    if j1.button("➕ Add this trade to journal", use_container_width=True, key="sm_add"):
        st.session_state[JOURNAL_KEY].append(new_row)
        st.success(f"Added {ticker} to journal (now {len(st.session_state[JOURNAL_KEY])} trade(s)).")
    if j2.button("🗑️ Clear journal", use_container_width=True, key="sm_clear"):
        st.session_state[JOURNAL_KEY] = []
        st.info("Journal cleared.")

    if st.session_state[JOURNAL_KEY]:
        journal_df = pd.DataFrame(st.session_state[JOURNAL_KEY])
        st.dataframe(journal_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download full journal (CSV)",
            data=journal_df.to_csv(index=False).encode("utf-8"),
            file_name="smart_money_trade_journal.csv", mime="text/csv",
        )
    else:
        # Always provide a single-row CSV even if user hasn't pressed Add
        st.download_button(
            "Download this trade plan only (CSV)",
            data=pd.DataFrame([new_row]).to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_smart_money_trade_plan.csv", mime="text/csv",
        )

    with st.expander("What this analyzer is doing"):
        st.markdown(
            """
            This is a **semi-automatic trade planner**, not a black-box signal bot.

            **Automatic:**
            - EMA alignment + rough trend context
            - Swing highs / lows
            - Body-close BOS / displacement checks
            - Potential demand / supply MB zones
            - Risk, R:R, 1R/2R/3R levels, and position size
            - Trade grading + journal export

            **You manually confirm:**
            - Whether the MB is truly valid
            - Whether the POI is worth trading
            - Whether the setup matches your ACCMB → MMB execution model
            - Whether Wyckoff is actually present inside the zone
            """
        )
