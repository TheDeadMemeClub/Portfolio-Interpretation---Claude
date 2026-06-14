"""Action Center — a single prioritized to-do list scanned across the whole book.

Pulls together signals that otherwise live in separate tabs so the answer to
"what needs my attention today?" is one glance:

  * allocation bands breached (per asset class)
  * single-name concentration (>10% of book)
  * live tax-loss-harvest opportunities
  * earnings inside a short window
  * IV-rich names (covered-call candidates: IV >> realized vol)
  * deep drawdowns from the 52-week high
  * bearish trend on a sizable position
  * cash drift vs target (too much dry powder / too little)

Pure function: returns a sorted DataFrame. The UI just renders it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_PRI_ORDER = {"🔴": 0, "🟡": 1, "🟢": 2}


def build_signals(
    summary: pd.DataFrame,
    allocation: pd.DataFrame,
    lots: pd.DataFrame,
    *,
    broker_total: float,
    targets: dict[str, float] | None = None,
    band_tolerance: float = 0.05,
    tlh_lots: pd.DataFrame | None = None,
    earnings: pd.DataFrame | None = None,
    iv_table: pd.DataFrame | None = None,
    earnings_window_days: int = 7,
    concentration_cap: float = 0.10,
    drawdown_floor: float = -0.25,
) -> pd.DataFrame:
    rows: list[dict] = []

    def add(pri, cat, sym, signal, action):
        rows.append({"Priority": pri, "Category": cat, "Symbol": sym,
                     "Signal": signal, "Suggested Action": action})

    # --- Allocation bands -------------------------------------------------
    if targets and broker_total:
        cur = allocation.set_index("Asset Class")["Market Value"].to_dict()
        for cls, tgt in targets.items():
            w = float(cur.get(cls, 0.0)) / broker_total
            drift = w - tgt
            if abs(drift) > band_tolerance:
                pri = "🔴" if abs(drift) > band_tolerance * 2 else "🟡"
                direction = "over" if drift > 0 else "under"
                add(pri, "Allocation", cls,
                    f"{cls} at {w:.0%} vs {tgt:.0%} target ({direction} by {abs(drift):.0%})",
                    f"{'Trim' if drift > 0 else 'Add to'} {cls} — see Rebalance tab")

    # --- Single-name concentration ----------------------------------------
    if "Portfolio Weight" in summary.columns:
        for _, r in summary.iterrows():
            w = float(r.get("Portfolio Weight", 0) or 0)
            if w > concentration_cap:
                pri = "🔴" if w > concentration_cap * 1.5 else "🟡"
                add(pri, "Concentration", r["Symbol"],
                    f"{r['Symbol']} is {w:.0%} of the book",
                    "Consider trimming to manage single-name risk")

    # --- Live TLH opportunities -------------------------------------------
    if tlh_lots is not None and not tlh_lots.empty:
        by_sym = (tlh_lots.groupby("Symbol")["Unrealized P&L $"].sum()
                  .sort_values())
        for sym, loss in by_sym.items():
            if loss < 0:
                add("🟡", "Tax", sym,
                    f"${abs(loss):,.0f} harvestable loss in {sym}",
                    "Harvest to offset gains — mind the 30-day wash-sale window")

    # --- Earnings soon ----------------------------------------------------
    if earnings is not None and not earnings.empty:
        e = earnings.copy()
        date_col = next((c for c in e.columns if "date" in c.lower()), None)
        if date_col:
            e["_dt"] = pd.to_datetime(e[date_col], errors="coerce")
            today = pd.Timestamp.today().normalize()
            soon = e[(e["_dt"] >= today)
                     & (e["_dt"] <= today + pd.Timedelta(days=earnings_window_days))]
            sym_col = next((c for c in e.columns if c.lower() in ("symbol", "ticker")), None)
            for _, r in soon.iterrows():
                d = (r["_dt"] - today).days
                sym = r.get(sym_col, "") if sym_col else ""
                add("🟡", "Earnings", sym,
                    f"{sym} reports in {d} day{'s' if d != 1 else ''}",
                    "Review exposure / option positioning into the print")

    # --- IV-rich (covered-call candidates) --------------------------------
    if iv_table is not None and not iv_table.empty:
        iv = iv_table.copy()
        sym_col = next((c for c in iv.columns if c.lower() in ("symbol", "ticker")), None)
        read_col = next((c for c in iv.columns if "read" in c.lower() or "signal" in c.lower()), None)
        for _, r in iv.iterrows():
            read = str(r.get(read_col, "")).lower() if read_col else ""
            if "rich" in read or "elevated" in read or "high" in read:
                sym = r.get(sym_col, "") if sym_col else ""
                add("🟢", "Options", sym,
                    f"{sym} options look IV-rich",
                    "Covered-call candidate — see Options tab")

    # --- Deep drawdowns ---------------------------------------------------
    if "Drawdown from 52W High %" in summary.columns:
        for _, r in summary.iterrows():
            dd = r.get("Drawdown from 52W High %")
            w = float(r.get("Portfolio Weight", 0) or 0)
            if pd.notna(dd) and dd < drawdown_floor and w > 0.01:
                add("🟡", "Risk", r["Symbol"],
                    f"{r['Symbol']} is {dd:.0%} off its 52-week high",
                    "Re-check the thesis — averaging down vs cutting")

    # --- Bearish trend on a sizable position ------------------------------
    if "Trend Rating" in summary.columns:
        for _, r in summary.iterrows():
            w = float(r.get("Portfolio Weight", 0) or 0)
            if str(r.get("Trend Rating", "")) == "Bearish" and w > 0.03:
                add("🟡", "Trend", r["Symbol"],
                    f"{r['Symbol']} ({w:.0%}) is in a bearish trend",
                    "Tighten risk or reassess conviction")

    # --- Cash drift -------------------------------------------------------
    if targets and broker_total:
        cash_w = float(allocation.set_index("Asset Class")["Market Value"]
                       .get("Cash", 0.0)) / broker_total
        cash_t = float(targets.get("Cash", 0.0))
        if cash_w - cash_t > band_tolerance:
            add("🟢", "Cash", "CASH",
                f"Cash at {cash_w:.0%} vs {cash_t:.0%} target — dry powder building",
                "Deploy on a dip or move toward target")
        elif cash_t - cash_w > band_tolerance:
            add("🟡", "Cash", "CASH",
                f"Cash at {cash_w:.0%} vs {cash_t:.0%} target — thin buffer",
                "Raise cash toward your buffer target")

    if not rows:
        return pd.DataFrame(columns=["Priority", "Category", "Symbol",
                                     "Signal", "Suggested Action"])
    df = pd.DataFrame(rows)
    df["_o"] = df["Priority"].map(_PRI_ORDER).fillna(3)
    return df.sort_values(["_o", "Category"]).drop(columns="_o").reset_index(drop=True)


def signal_counts(signals: pd.DataFrame) -> dict:
    if signals is None or signals.empty:
        return {"🔴": 0, "🟡": 0, "🟢": 0, "total": 0}
    vc = signals["Priority"].value_counts().to_dict()
    return {"🔴": vc.get("🔴", 0), "🟡": vc.get("🟡", 0),
            "🟢": vc.get("🟢", 0), "total": len(signals)}
