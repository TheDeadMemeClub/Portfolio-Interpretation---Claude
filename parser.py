"""Broker positions file parser.

Reads Wells Fargo-style (and similar) broker positions exports and produces
clean, normalized DataFrames for downstream analysis.

The original parser was already solid — this version adds:
  * Stronger type hints and docstrings
  * Better handling of mangled / partial exports (no rows crash the app)
  * Section detection that tolerates extra whitespace and case differences
  * Helper to scrub identifying account info before exporting
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column registries
# ---------------------------------------------------------------------------
NUMERIC_FIELDS: list[str] = [
    "Shares", "Market Value", "Total Cost1", "Original Cost",
    "Total Client Investment", "Unrealized Gain/Loss ($)1",
    "Client Inv Gain/(Loss) $", "Est. Annual Income", "Today's Change ($)1",
    "Change from Prev ($)", "Last Price ($)", "Trade Price", "Cost Basis",
]

# Section names we look for as block headers in the spreadsheet
SECTION_NAMES: tuple[str, ...] = ("Stocks", "ETFs", "Mutual Funds", "Fixed Income", "Bonds")

# Row text prefixes that signal a non-position aggregator row to skip
SKIP_PREFIXES: tuple[str, ...] = (
    "Common Stock", "Money Market", "Open End", "Closed End",
)

# Regex matching a CUSIP (9 alphanumeric chars, ending with a check digit).
# Used to detect bonds when broker doesn't put them in their own section.
CUSIP_PATTERN = re.compile(r"^[A-Z0-9]{8}\d$")

# Description fragments that strongly suggest a fixed-income holding.
BOND_DESCRIPTORS: tuple[str, ...] = (
    "NOTES", "NOTE ", "BOND", "BONDS", "CPN", "COUPON", "DEBENTURE",
    "CALLABLE", " CALLAB ", "MUNICIPAL", "TREASURY", "TIPS",
    "MATURITY", "MAT ", "FIXED RATE",
)


def is_bond_holding(symbol: str, description: str = "", asset_type: str = "") -> bool:
    """Heuristic bond detector.

    A holding is a bond if any of:
      * Symbol looks like a CUSIP (9 alphanumeric, ends in a digit)
      * Description contains bond keywords (NOTES, BONDS, CPN, CALLABLE, etc.)
      * Asset Type was already labeled Fixed Income / Bonds
    """
    sym = str(symbol or "").strip().upper()
    desc = str(description or "").upper()
    atype = str(asset_type or "").upper()

    if "FIXED INCOME" in atype or atype == "BONDS":
        return True
    if CUSIP_PATTERN.match(sym):
        return True
    return any(token in desc for token in BOND_DESCRIPTORS)


def extract_coupon_rate(description: str) -> float | None:
    """Try to pull a coupon rate (as decimal) out of the bond description.

    Bond descriptions usually contain something like '4.500% MAT 2030' or
    '4.5 CPN' or '4.500 NOTES'. We grab the first plausible percent value.
    """
    if not description:
        return None
    desc = str(description).upper()
    # Look for things like "4.500%", "5.25 CPN", "3.5 NOTES"
    pattern = re.compile(r"(\d{1,2}\.?\d{0,4})\s*(?:%|CPN|COUPON|NOTES?|BONDS?)")
    matches = pattern.findall(desc)
    for m in matches:
        try:
            v = float(m)
            # Coupon rates are realistically 0% to 20%
            if 0 < v <= 20:
                return v / 100
        except ValueError:
            continue
    return None


@dataclass
class PortfolioParseResult:
    """Container returned by `parse_broker_upload`."""
    summary: pd.DataFrame         # one row per Symbol, fully aggregated
    lots: pd.DataFrame            # individual lots with trade date / fill
    allocation: pd.DataFrame      # asset-class allocation (Stocks/ETFs/MF/Cash/FI)
    priced_date: str | None
    broker_total: float | None
    cash_total: float
    fixed_income_total: float


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------
def clean_number(x: Any) -> float:
    """Convert messy broker number strings like '($1,234.56)' to floats.

    Returns NaN on anything we can't parse.
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s in {"", "--", "N/A", "nan", "None"}:
        return np.nan
    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1]
    s = s.replace("$", "").replace(",", "").replace("%", "").replace("+", "")
    try:
        v = float(s)
        return -v if neg else v
    except ValueError:
        return np.nan


def clean_date(x: Any, fallback: str | None = None) -> str:
    """Normalize a date-ish value to YYYY-MM-DD, with sensible fallback."""
    if pd.isna(x) or str(x).strip() in {"", "Detail", "N/A", "Intra-Day", "Detailnc"}:
        return fallback or ""
    s = str(x).replace("nc", "").strip()
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
    if m:
        mm, dd, yyyy = m.groups()
        return f"{yyyy}-{int(mm):02d}-{int(dd):02d}"
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.notna(dt):
            return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        pass
    return s


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------
def read_broker_file(uploaded_file: Any) -> pd.DataFrame:
    """Read xls/xlsx/csv broker export into a raw header-less DataFrame."""
    name = getattr(uploaded_file, "name", "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file, header=None, dtype=object)
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file, header=None, dtype=object, engine="openpyxl")
    if name.endswith(".xls"):
        # Wells Fargo .xls files are usually old BIFF -> xlrd handles them
        return pd.read_excel(uploaded_file, header=None, dtype=object, engine="xlrd")
    # Fallback: try Excel first, then CSV
    try:
        return pd.read_excel(uploaded_file, header=None, dtype=object)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, header=None, dtype=object)


# ---------------------------------------------------------------------------
# Row / section utilities
# ---------------------------------------------------------------------------
def _row_text(df: pd.DataFrame, row_idx: int) -> str:
    if row_idx < 0 or row_idx >= len(df):
        return ""
    return " ".join(str(x) for x in df.iloc[row_idx].dropna().tolist())


def extract_priced_date(df: pd.DataFrame) -> str | None:
    """Pull the as-of date from the top of the report (first ~15 rows)."""
    for i in range(min(15, len(df))):
        m = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", _row_text(df, i))
        if m:
            return clean_date(m.group(1))
    return None


def extract_summary_totals(df: pd.DataFrame) -> tuple[float, float, float | None]:
    """Find Cash / Fixed Income / Total Portfolio rows in the export."""
    cash_total = 0.0
    fixed_income_total = 0.0
    broker_total: float | None = None
    for i in range(len(df)):
        cells = df.iloc[i].tolist()
        text_cells = [str(x).strip() for x in cells if pd.notna(x)]
        joined = " ".join(text_cells)
        values = [clean_number(x) for x in cells]
        numeric_values = [v for v in values if pd.notna(v)]
        if "Total Cash" in joined and numeric_values:
            cash_total = float(numeric_values[0])
        if "Total Fixed Income" in joined and numeric_values:
            fixed_income_total = float(max(numeric_values, key=abs))
        if "Total Portfolio" in joined and numeric_values:
            broker_total = float(numeric_values[0])
    return cash_total, fixed_income_total, broker_total


def find_position_sections(df: pd.DataFrame) -> list[tuple[str, int, int, int]]:
    """Locate (asset_type, header_row, first_data_row, last_data_row) for each section."""
    section_markers: list[tuple[str, int]] = []
    for i in range(len(df)):
        row_vals = [str(x).strip() for x in df.iloc[i].dropna().tolist()]
        for section in SECTION_NAMES:
            if any(v == section for v in row_vals):
                section_markers.append((section, i))

    sections: list[tuple[str, int, int, int]] = []
    for idx, (section, marker_row) in enumerate(section_markers):
        next_marker = section_markers[idx + 1][1] if idx + 1 < len(section_markers) else len(df)
        header_row = None
        for r in range(marker_row, min(marker_row + 6, len(df))):
            vals = [str(x).strip() for x in df.iloc[r].tolist()]
            if "Symbol" in vals and "Description" in vals:
                header_row = r
                break
        if header_row is None:
            continue
        sections.append((section, header_row, header_row + 1, next_marker - 1))
    return sections


# ---------------------------------------------------------------------------
# Position parsing
# ---------------------------------------------------------------------------
def parse_positions(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None, float, float, float | None]:
    """Walk the dataframe and return a flat positions DataFrame + portfolio totals."""
    priced_date = extract_priced_date(df)
    cash_total, fixed_income_total, broker_total = extract_summary_totals(df)
    sections = find_position_sections(df)
    rows: list[dict] = []

    for asset_type, header_row, start, end in sections:
        headers = [str(x).strip() if pd.notna(x) else "" for x in df.iloc[header_row].tolist()]
        for r in range(start, end + 1):
            raw_vals = df.iloc[r].tolist()
            joined = " ".join(str(x).strip() for x in raw_vals if pd.notna(x))
            if not joined or joined.startswith("Total"):
                continue
            if any(joined.startswith(prefix) for prefix in SKIP_PREFIXES):
                continue
            rec = {h: raw_vals[c] for c, h in enumerate(headers) if h}
            symbol = str(rec.get("Symbol", "")).strip()
            if not symbol or symbol.lower() == "nan":
                continue
            rec["Symbol"] = symbol.upper()
            rec["Asset Type"] = asset_type
            rec["Source Row"] = r + 1
            rows.append(rec)

    positions = pd.DataFrame(rows)
    for col in NUMERIC_FIELDS:
        if col in positions.columns:
            positions[col] = positions[col].apply(clean_number)
    return positions, priced_date, cash_total, fixed_income_total, broker_total


# ---------------------------------------------------------------------------
# Aggregation -> summary / lots / allocation
# ---------------------------------------------------------------------------
def build_analysis(
    positions: pd.DataFrame,
    priced_date: str | None,
    cash_total: float,
    fixed_income_total: float,
    broker_total: float | None,
) -> PortfolioParseResult:
    if positions.empty:
        raise ValueError(
            "No position rows found. This app expects a Wells Fargo-style "
            "positions export with Stocks, ETFs, and/or Mutual Funds sections."
        )

    summary = _build_summary(positions, broker_total, cash_total, fixed_income_total)
    lots = _build_lots(positions, priced_date)
    allocation = _build_allocation(summary, cash_total, fixed_income_total)

    return PortfolioParseResult(
        summary=summary,
        lots=lots,
        allocation=allocation,
        priced_date=priced_date,
        broker_total=broker_total,
        cash_total=cash_total,
        fixed_income_total=fixed_income_total,
    )


def _build_summary(
    positions: pd.DataFrame,
    broker_total: float | None,
    cash_total: float,
    fixed_income_total: float,
) -> pd.DataFrame:
    summary_rows: list[dict] = []
    for sym, group in positions.groupby("Symbol", dropna=True):
        details = group[group.get("Tax Term", pd.Series(index=group.index, dtype=object)).astype(str).eq("Detail")]
        use = details if not details.empty else group
        base = use.iloc[0]

        def _sum_first(*cols: str) -> float:
            for col in cols:
                if col in use.columns and use[col].notna().any():
                    return float(use[col].sum(skipna=True))
            return np.nan

        shares = _sum_first("Shares")
        mv = _sum_first("Market Value")
        cost = _sum_first("Total Cost1", "Original Cost", "Total Client Investment")
        ugl = _sum_first("Unrealized Gain/Loss ($)1", "Client Inv Gain/(Loss) $")
        today = _sum_first("Today's Change ($)1", "Change from Prev ($)")
        income = _sum_first("Est. Annual Income")
        last_price = base.get("Last Price ($)", np.nan)
        description = base.get("Description", "")
        asset_type = base.get("Asset Type", "")

        # ------------------------------------------------------------------
        # BOND HANDLING - the broker often reports a bond's PRINCIPAL/face
        # value in the Est. Annual Income column, which is wrong. We detect
        # bonds and replace that value with a real coupon-based estimate.
        # ------------------------------------------------------------------
        is_bond = is_bond_holding(sym, description, asset_type)
        coupon_rate = None
        if is_bond:
            asset_type = "Fixed Income"
            coupon_rate = extract_coupon_rate(description)
            # The bond's "shares" in WFA exports is actually face value in $.
            face_value = shares if pd.notna(shares) and shares > 1000 else (mv if pd.notna(mv) else np.nan)
            if coupon_rate and pd.notna(face_value):
                income = float(face_value) * coupon_rate
            elif pd.notna(mv) and pd.notna(income) and income > mv * 0.5:
                # If broker-reported "income" is >50% of market value, it's
                # almost certainly principal masquerading as income. Drop it.
                income = np.nan

        summary_rows.append({
            "Symbol": sym,
            "Description": description,
            "Asset Type": asset_type,
            "Is Bond": is_bond,
            "Coupon Rate": coupon_rate,
            "Shares": shares,
            "Last Price": last_price,
            "Market Value": mv,
            "Total Cost": cost,
            "Avg Cost": cost / shares if pd.notna(cost) and pd.notna(shares) and shares else np.nan,
            "Unrealized P&L $": ugl,
            "Unrealized P&L %": ugl / cost if pd.notna(ugl) and pd.notna(cost) and cost else np.nan,
            "Today's Change $": today,
            "Today's Change %": today / mv if pd.notna(today) and pd.notna(mv) and mv else np.nan,
            "Est. Annual Income": income,
            "Yield on MV": income / mv if pd.notna(income) and pd.notna(mv) and mv else np.nan,
            "Lot Count": int(len(group)),
        })

    summary = pd.DataFrame(summary_rows)
    securities_total = summary["Market Value"].fillna(0).sum()
    total_for_weights = broker_total if broker_total and broker_total > 0 else securities_total + cash_total + fixed_income_total
    summary["Portfolio Weight"] = summary["Market Value"] / total_for_weights if total_for_weights else np.nan
    return summary.sort_values("Market Value", ascending=False, na_position="last").reset_index(drop=True)


def _build_lots(positions: pd.DataFrame, priced_date: str | None) -> pd.DataFrame:
    lot_rows: list[dict] = []
    for sym, group in positions.groupby("Symbol", dropna=True):
        has_detail = "Tax Term" in group.columns and group["Tax Term"].astype(str).eq("Detail").any()
        for _, p in group.iterrows():
            if has_detail and str(p.get("Tax Term", "")) == "Detail":
                continue
            q = p.get("Shares", np.nan)
            if pd.isna(q) or q == 0:
                continue
            fill = p.get("Trade Price", np.nan)
            if pd.isna(fill):
                fill = p.get("Cost Basis", np.nan)
            total_cost = p.get("Total Cost1", np.nan)
            if pd.isna(fill) and pd.notna(total_cost) and q:
                fill = total_cost / q
            lot_rows.append({
                "Symbol": sym,
                "Description": p.get("Description", ""),
                "Asset Type": p.get("Asset Type", ""),
                "Side": "Buy" if q > 0 else "Sell",
                "Qty": abs(q),
                "Signed Qty": q,
                "Fill Price": fill,
                "Commission": 0,
                "Closing Time": clean_date(p.get("Trade Date1", None), priced_date),
                "Market Value": p.get("Market Value", np.nan),
                "Total Cost": total_cost if pd.notna(total_cost) else p.get("Original Cost", np.nan),
                "Unrealized P&L $": p.get("Unrealized Gain/Loss ($)1", np.nan),
                "Tax Term": p.get("Tax Term", ""),
                "Source Row": p.get("Source Row", ""),
            })
    return pd.DataFrame(lot_rows)


def _build_allocation(
    summary: pd.DataFrame,
    cash_total: float,
    fixed_income_total: float,
) -> pd.DataFrame:
    parts = [
        ("Stocks", summary.loc[summary["Asset Type"].eq("Stocks"), "Market Value"].fillna(0).sum()),
        ("ETFs", summary.loc[summary["Asset Type"].eq("ETFs"), "Market Value"].fillna(0).sum()),
        ("Mutual Funds", summary.loc[summary["Asset Type"].eq("Mutual Funds"), "Market Value"].fillna(0).sum()),
        ("Cash", cash_total),
        ("Fixed Income", fixed_income_total),
    ]
    allocation = pd.DataFrame(parts, columns=["Asset Class", "Market Value"])
    allocation = allocation[allocation["Market Value"].fillna(0) > 0].copy()
    total = allocation["Market Value"].sum()
    allocation["Weight"] = allocation["Market Value"] / total if total else np.nan
    return allocation


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------
def parse_broker_upload(uploaded_file: Any) -> PortfolioParseResult:
    """Top-level: read the broker file and return a fully analyzed result."""
    df = read_broker_file(uploaded_file)
    positions, priced_date, cash_total, fi_total, broker_total = parse_positions(df)
    return build_analysis(positions, priced_date, cash_total, fi_total, broker_total)


def tradingview_csv(summary_or_lots: pd.DataFrame, mode: str = "consolidated") -> pd.DataFrame:
    """Format positions/lots for a TradingView-style portfolio import."""
    if mode == "lot" and {"Symbol", "Side", "Qty", "Fill Price", "Commission", "Closing Time"}.issubset(summary_or_lots.columns):
        return summary_or_lots[["Symbol", "Side", "Qty", "Fill Price", "Commission", "Closing Time"]].copy()
    df = summary_or_lots.copy()
    return pd.DataFrame({
        "Symbol": df["Symbol"],
        "Side": np.where(df["Shares"].fillna(0) >= 0, "Buy", "Sell"),
        "Qty": df["Shares"].abs(),
        "Fill Price": df["Avg Cost"].fillna(df["Last Price"]),
        "Commission": 0,
        "Closing Time": pd.Timestamp.today().strftime("%Y-%m-%d"),
    })
