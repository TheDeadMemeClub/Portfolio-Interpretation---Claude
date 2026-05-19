"""Export utilities: CSV, ZIP, PDF.

NEW module — the original app only had ad-hoc CSV download buttons.
This consolidates exports and adds:
  * A polished PDF summary (one-page advisor cockpit + tables)
  * A combined ZIP bundle of all CSVs

PDF generation uses fpdf2 (pure Python, no system fonts required).
If fpdf2 isn't installed, the PDF function returns None and the UI
hides the button — nothing crashes.
"""
from __future__ import annotations

import io
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from fpdf import FPDF
    _HAS_FPDF = True
except Exception:
    _HAS_FPDF = False


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# ZIP bundle
# ---------------------------------------------------------------------------
def build_zip(files: dict[str, bytes]) -> bytes:
    """Build a ZIP from a {filename: bytes} mapping."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            if data:
                z.writestr(name, data)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# PDF summary
# ---------------------------------------------------------------------------
class _PortfolioPDF(FPDF if _HAS_FPDF else object):  # type: ignore[misc]
    def header(self):  # type: ignore[override]
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(45, 45, 55)
        self.cell(0, 8, "Portfolio EPIC — dashboard summary", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(110, 110, 120)
        self.cell(0, 5, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                 new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):  # type: ignore[override]
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 160)
        self.cell(0, 5, f"Page {self.page_no()}", align="C")


def _add_section(pdf: "_PortfolioPDF", title: str) -> None:
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(40, 40, 50)
    pdf.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(220, 220, 230)
    pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
    pdf.ln(1)


def _add_kv_row(pdf: "_PortfolioPDF", label: str, value: str) -> None:
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(80, 80, 90)
    pdf.cell(60, 5, label)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(30, 30, 40)
    pdf.cell(0, 5, value, new_x="LMARGIN", new_y="NEXT")


def _add_table(pdf: "_PortfolioPDF", df: pd.DataFrame, col_widths: list[float]) -> None:
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(245, 245, 250)
    pdf.set_text_color(40, 40, 50)
    for col, w in zip(df.columns, col_widths):
        pdf.cell(w, 5, str(col)[:22], border=0, fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(60, 60, 70)
    for _, row in df.iterrows():
        for col, w in zip(df.columns, col_widths):
            val = row[col]
            if isinstance(val, float):
                txt = f"{val:,.2f}" if abs(val) >= 1 else f"{val:.4f}"
            else:
                txt = str(val) if pd.notna(val) else ""
            pdf.cell(w, 5, txt[:22])
        pdf.ln()


def build_pdf_report(
    *,
    broker_total: float,
    ugl_total: float,
    today_total: float,
    income_total: float,
    cash_weight: float,
    top1_weight: float,
    top5_weight: float,
    flag_items: list[str],
    allocation: pd.DataFrame,
    top_holdings: pd.DataFrame,
    sector_breakdown: pd.DataFrame | None = None,
    tlh_candidates: pd.DataFrame | None = None,
) -> bytes | None:
    """Build the one-page (ish) PDF summary. Returns None if fpdf2 isn't available."""
    if not _HAS_FPDF:
        return None

    pdf = _PortfolioPDF()
    pdf.add_page()

    # Headline metrics block
    _add_section(pdf, "Headline metrics")
    _add_kv_row(pdf, "Total portfolio value", f"${broker_total:,.0f}")
    _add_kv_row(pdf, "Unrealized P&L",
                f"${ugl_total:,.0f}  ({(ugl_total/broker_total if broker_total else 0):+.2%})")
    _add_kv_row(pdf, "Today's change",
                f"${today_total:,.0f}  ({(today_total/broker_total if broker_total else 0):+.2%})")
    _add_kv_row(pdf, "Estimated annual income", f"${income_total:,.0f}")
    _add_kv_row(pdf, "Cash weight", f"{cash_weight:.1%}")
    _add_kv_row(pdf, "Largest position weight", f"{top1_weight:.1%}")
    _add_kv_row(pdf, "Top-5 weight", f"{top5_weight:.1%}")

    # Advisor cockpit flags
    _add_section(pdf, "Advisor cockpit — items needing attention")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(60, 60, 70)
    for item in flag_items:
        pdf.multi_cell(0, 5, f"• {item}")

    # Allocation
    if allocation is not None and not allocation.empty:
        _add_section(pdf, "Asset allocation")
        alloc = allocation.copy()
        alloc["Market Value"] = alloc["Market Value"].map(lambda x: f"${x:,.0f}")
        alloc["Weight"] = alloc["Weight"].map(lambda x: f"{x:.1%}")
        _add_table(pdf, alloc[["Asset Class", "Market Value", "Weight"]], [60, 60, 40])

    # Top holdings
    if top_holdings is not None and not top_holdings.empty:
        _add_section(pdf, "Top holdings")
        cols = [c for c in ["Symbol", "Market Value", "Portfolio Weight",
                            "Unrealized P&L $", "Unrealized P&L %",
                            "Trend Rating", "Risk Tier"] if c in top_holdings.columns]
        th = top_holdings[cols].head(10).copy()
        if "Market Value" in th: th["Market Value"] = th["Market Value"].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
        if "Portfolio Weight" in th: th["Portfolio Weight"] = th["Portfolio Weight"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
        if "Unrealized P&L $" in th: th["Unrealized P&L $"] = th["Unrealized P&L $"].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
        if "Unrealized P&L %" in th: th["Unrealized P&L %"] = th["Unrealized P&L %"].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "")
        widths = [22, 32, 22, 28, 22, 28, 22][:len(cols)]
        _add_table(pdf, th, widths)

    # Sector breakdown
    if sector_breakdown is not None and not sector_breakdown.empty:
        _add_section(pdf, "Sector exposure")
        sb = sector_breakdown[["Sector", "Portfolio Weight", "Positions"]].head(10).copy()
        sb["Portfolio Weight"] = sb["Portfolio Weight"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
        _add_table(pdf, sb, [80, 50, 30])

    # TLH candidates
    if tlh_candidates is not None and not tlh_candidates.empty:
        flagged = tlh_candidates[tlh_candidates.get("TLH Candidate", "") == "✓"].head(10)
        if not flagged.empty:
            _add_section(pdf, "Tax-loss harvesting candidates")
            cols = [c for c in ["Symbol", "Qty", "Unrealized P&L $", "Unrealized P&L %", "Tax Term"] if c in flagged.columns]
            tt = flagged[cols].copy()
            if "Qty" in tt: tt["Qty"] = tt["Qty"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
            if "Unrealized P&L $" in tt: tt["Unrealized P&L $"] = tt["Unrealized P&L $"].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
            if "Unrealized P&L %" in tt: tt["Unrealized P&L %"] = tt["Unrealized P&L %"].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "")
            _add_table(pdf, tt, [25, 25, 40, 30, 40][:len(cols)])

    out = pdf.output(dest="S")
    if isinstance(out, str):
        out = out.encode("latin-1", "ignore")
    return bytes(out)


def pdf_available() -> bool:
    return _HAS_FPDF
