"""Tax-aware rebalance trade generator.

Fuses target-band rebalancing, tax-lot selection, and wash-sale awareness into a
single actionable blotter. Pure functions only (no Streamlit) so the logic is
unit-testable and validated against portfolio anchors before the UI trusts it.

Lot-selection philosophy
------------------------
When an asset class is overweight we need to raise cash by selling lots. We pick
which lots to sell in *tax-cost* order, cheapest first:

  1. Loss lots          -> harvest the loss (negative tax = a SAVING)
  2. Long-term gains    -> taxed at the LT cap-gains rate
  3. Short-term gains   -> taxed at the ordinary rate (worst, sold last)

This mirrors the principle that harvesting loss lots first can offset gains from
trimming appreciated positions, often turning a tax bill into a net saving.

Wash-sale guard
---------------
The IRS disallows a loss if a substantially identical security is bought within
30 days before *or* after the sale. We flag any loss-lot sale where the same
symbol was acquired in the last 30 days (or is slated for a rebuy), and offer a
toggle to exclude those sales entirely.

Nothing here is tax advice — it is an estimate. Real netting (ST/LT offset, the
$3k ordinary-income allowance, and carryforwards) is more nuanced, so the engine
also returns raw realized gain/loss by term for exact downstream computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Asset classes that hold sellable lots (Cash / Fixed Income are handled as
# residual funding, not lot-by-lot trims).
_SELLABLE_CLASSES = ("Stocks", "ETFs", "Mutual Funds")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_term(tax_term: object, days_held: float | None) -> str:
    """Map a messy broker 'Tax Term' string to 'LT' / 'ST'.

    Falls back to the 365-day rule using days held when the broker left the term
    blank (common for intra-day or freshly transferred lots).
    """
    s = str(tax_term).strip().lower()
    if "long" in s or s in {"lt", "l/t"}:
        return "LT"
    if "short" in s or s in {"st", "s/t"}:
        return "ST"
    if days_held is not None and not pd.isna(days_held):
        return "LT" if days_held > 365 else "ST"
    # Unknown -> treat as ST (most conservative tax assumption).
    return "ST"


def _days_held(closing_time: object, as_of: pd.Timestamp) -> float:
    try:
        dt = pd.to_datetime(closing_time, errors="coerce")
        if pd.isna(dt):
            return np.nan
        return float((as_of - dt).days)
    except Exception:
        return np.nan


def prepare_lots(lots: pd.DataFrame, *, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Enrich raw lots with per-share value, days held, normalized term, and a
    tax-cost-per-dollar key used to order sales tax-efficiently."""
    if lots is None or lots.empty:
        return pd.DataFrame()

    as_of = as_of or pd.Timestamp.today().normalize()
    df = lots.copy()

    # Only equity-like lots are tradable here.
    df = df[df["Asset Type"].isin(_SELLABLE_CLASSES)].copy()
    df = df[df.get("Side", "Buy").astype(str).str.lower().ne("sell")]
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce")
    df = df[df["Qty"].fillna(0) > 0].copy()
    if df.empty:
        return df

    df["Market Value"] = pd.to_numeric(df["Market Value"], errors="coerce")
    df["Total Cost"] = pd.to_numeric(df["Total Cost"], errors="coerce")
    df["Unrealized P&L $"] = pd.to_numeric(df["Unrealized P&L $"], errors="coerce")

    # If broker UGL is missing, derive from MV - cost.
    df["Unrealized P&L $"] = df["Unrealized P&L $"].fillna(
        df["Market Value"] - df["Total Cost"]
    )
    # If MV missing but we have cost + UGL, reconstruct (fallback for stale lots).
    df["Market Value"] = df["Market Value"].fillna(
        df["Total Cost"] + df["Unrealized P&L $"]
    )

    df["Price/Share"] = np.where(
        df["Qty"] > 0, df["Market Value"] / df["Qty"], np.nan
    )
    df["Days Held"] = df["Closing Time"].apply(lambda x: _days_held(x, as_of))
    df["Term"] = [
        normalize_term(t, d) for t, d in zip(df["Tax Term"], df["Days Held"])
    ]

    # Tax-cost ranking: lower = sell first.
    #   loss lots          -> negative
    #   long-term gains     -> small positive
    #   short-term gains    -> large positive
    gain_per_dollar = np.where(
        df["Market Value"].abs() > 0,
        df["Unrealized P&L $"] / df["Market Value"].abs(),
        0.0,
    )
    term_penalty = np.where(df["Term"].eq("ST"), 1.0, 0.5)
    df["_sell_rank"] = np.where(
        df["Unrealized P&L $"] < 0,
        df["Unrealized P&L $"] / df["Market Value"].abs(),  # negative -> first
        gain_per_dollar * term_penalty,                     # gains -> later
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class RebalancePlan:
    trades: pd.DataFrame                 # the blotter
    post_allocation: pd.DataFrame        # resulting weights vs target
    stats: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------
def generate_plan(
    allocation: pd.DataFrame,
    lots: pd.DataFrame,
    summary: pd.DataFrame,
    targets: dict[str, float],
    broker_total: float,
    *,
    band_tolerance: float = 0.03,
    st_rate: float = 0.37,
    lt_rate: float = 0.20,
    avoid_wash_sales: bool = True,
    rebuy_symbols: tuple[str, ...] = (),
    as_of: pd.Timestamp | None = None,
) -> RebalancePlan:
    """Build a tax-aware rebalance blotter.

    Parameters
    ----------
    allocation : asset-class table (Asset Class, Market Value)
    lots       : per-lot table from the parser
    summary    : per-symbol table (for pro-rata buy suggestions + weights)
    targets    : {asset class -> target weight (0..1)}
    broker_total : total portfolio value (the reconciliation anchor)
    band_tolerance : only act on a class if it drifts beyond +/- this band
    st_rate, lt_rate : marginal short- and long-term tax rates
    avoid_wash_sales : drop loss-lot sales that would trip the 30-day rule
    rebuy_symbols : symbols you intend to buy (extends the wash-sale guard)
    """
    warnings: list[str] = []
    if broker_total is None or broker_total <= 0:
        broker_total = float(allocation["Market Value"].fillna(0).sum())

    alloc = allocation.set_index("Asset Class")["Market Value"].to_dict()
    prepped = prepare_lots(lots, as_of=as_of)

    # ---- 1. Compute target deltas by asset class -------------------------
    classes = sorted(set(list(alloc.keys()) + list(targets.keys())))
    drift_rows = []
    sell_need: dict[str, float] = {}
    buy_need: dict[str, float] = {}
    for cls in classes:
        cur_val = float(alloc.get(cls, 0.0))
        cur_w = cur_val / broker_total if broker_total else 0.0
        tgt_w = float(targets.get(cls, 0.0))
        tgt_val = tgt_w * broker_total
        delta = tgt_val - cur_val
        drift = cur_w - tgt_w
        in_band = abs(drift) <= band_tolerance
        drift_rows.append(
            {"Asset Class": cls, "Current Weight": cur_w, "Target Weight": tgt_w,
             "Drift": drift, "Delta $": delta, "In Band": in_band}
        )
        if in_band:
            continue
        if delta < 0 and cls in _SELLABLE_CLASSES:
            sell_need[cls] = -delta            # dollars to raise
        elif delta > 0:
            buy_need[cls] = delta              # dollars to deploy

    # ---- 2. Select lots to SELL (tax-efficient, wash-sale aware) ----------
    recent_buy_syms = set()
    if not prepped.empty:
        recent_buy_syms = set(
            prepped.loc[prepped["Days Held"] <= 30, "Symbol"].dropna().tolist()
        )
    rebuy_set = {s.upper() for s in rebuy_symbols} | {s.upper() for s in recent_buy_syms}

    trade_rows: list[dict] = []
    for cls, need in sell_need.items():
        pool = prepped[prepped["Asset Type"].eq(cls)].sort_values("_sell_rank")
        raised = 0.0
        for _, lot in pool.iterrows():
            if raised >= need:
                break
            sym = str(lot["Symbol"])
            is_loss = lot["Unrealized P&L $"] < 0
            wash_risk = is_loss and sym.upper() in rebuy_set
            if avoid_wash_sales and wash_risk:
                warnings.append(
                    f"Skipped loss sale of {sym} (lot {lot['Closing Time']}) — "
                    f"wash-sale risk (bought within 30 days / flagged for rebuy)."
                )
                continue

            remaining = need - raised
            lot_mv = float(lot["Market Value"]) if pd.notna(lot["Market Value"]) else 0.0
            if lot_mv <= 0:
                continue
            # Sell whole lot, or a partial slice if it more than covers the need.
            if lot_mv <= remaining * 1.0001:
                sell_frac = 1.0
            else:
                sell_frac = remaining / lot_mv
            shares = float(lot["Qty"]) * sell_frac
            proceeds = lot_mv * sell_frac
            realized = float(lot["Unrealized P&L $"]) * sell_frac
            term = lot["Term"]
            tax = realized * (st_rate if term == "ST" else lt_rate)  # neg if loss
            trade_rows.append({
                "Action": "SELL",
                "Asset Class": cls,
                "Symbol": sym,
                "Lot Date": lot["Closing Time"],
                "Shares": round(shares, 4),
                "Est. $": proceeds,
                "Realized G/L $": realized,
                "Term": term,
                "Est. Tax $": tax,
                "Wash-Sale": "⚠️" if wash_risk else "",
                "Rationale": (
                    "Harvest loss" if is_loss
                    else ("Trim — LT gain" if term == "LT" else "Trim — ST gain")
                ),
            })
            raised += proceeds
        if raised < need * 0.999:
            warnings.append(
                f"{cls}: only raised ${raised:,.0f} of the ${need:,.0f} needed "
                f"(not enough tradable lots after filters)."
            )

    proceeds_total = sum(t["Est. $"] for t in trade_rows if t["Action"] == "SELL")

    # ---- 3. Deploy proceeds + free cash into UNDERWEIGHT classes ----------
    # Cash already on hand that the plan wants to put to work = overweight cash.
    cash_cur = float(alloc.get("Cash", 0.0))
    cash_tgt = float(targets.get("Cash", 0.0)) * broker_total
    deployable_cash = max(0.0, cash_cur - cash_tgt)  # excess dry powder
    total_to_deploy = proceeds_total + deployable_cash

    buy_total_needed = sum(buy_need.values())
    if buy_total_needed > 0 and total_to_deploy > 0:
        for cls, need in buy_need.items():
            if cls == "Cash":
                continue  # raising cash is handled by *not* deploying it
            scale = min(1.0, total_to_deploy / buy_total_needed) if buy_total_needed else 0
            buy_amt = need * scale
            if buy_amt <= 0:
                continue
            # Pro-rata across existing holdings in that class (top up what you own).
            in_cls = summary[summary["Asset Type"].eq(cls)].copy()
            in_cls = in_cls[in_cls["Market Value"].fillna(0) > 0]
            if not in_cls.empty:
                w = in_cls["Market Value"] / in_cls["Market Value"].sum()
                for _, r in in_cls.iterrows():
                    amt = buy_amt * float(w.loc[r.name])
                    px = float(r["Last Price"]) if pd.notna(r.get("Last Price")) else np.nan
                    trade_rows.append({
                        "Action": "BUY",
                        "Asset Class": cls,
                        "Symbol": r["Symbol"],
                        "Lot Date": "",
                        "Shares": round(amt / px, 4) if px and not pd.isna(px) else np.nan,
                        "Est. $": amt,
                        "Realized G/L $": 0.0,
                        "Term": "",
                        "Est. Tax $": 0.0,
                        "Wash-Sale": "",
                        "Rationale": f"Add to {cls} (underweight)",
                    })
            else:
                trade_rows.append({
                    "Action": "BUY", "Asset Class": cls, "Symbol": "(your pick)",
                    "Lot Date": "", "Shares": np.nan, "Est. $": buy_amt,
                    "Realized G/L $": 0.0, "Term": "", "Est. Tax $": 0.0,
                    "Wash-Sale": "", "Rationale": f"New exposure — {cls} underweight",
                })

    trades = pd.DataFrame(trade_rows)

    # ---- 4. Stats + post-trade allocation --------------------------------
    realized_total = sum(t["Realized G/L $"] for t in trade_rows)
    realized_gains = sum(t["Realized G/L $"] for t in trade_rows if t["Realized G/L $"] > 0)
    realized_losses = sum(t["Realized G/L $"] for t in trade_rows if t["Realized G/L $"] < 0)
    st_net = sum(t["Realized G/L $"] for t in trade_rows if t["Term"] == "ST")
    lt_net = sum(t["Realized G/L $"] for t in trade_rows if t["Term"] == "LT")
    # Naive tax estimate: positive net per bucket taxed at its rate.
    est_tax = max(0.0, st_net) * st_rate + max(0.0, lt_net) * lt_rate
    # Tax that would've been due if we'd trimmed gains only (no harvesting).
    gains_only_tax = (
        sum(t["Realized G/L $"] for t in trade_rows
            if t["Realized G/L $"] > 0 and t["Term"] == "ST") * st_rate
        + sum(t["Realized G/L $"] for t in trade_rows
              if t["Realized G/L $"] > 0 and t["Term"] == "LT") * lt_rate
    )
    tax_saved = gains_only_tax - est_tax

    # Recompute allocation after applying class-level deltas actually traded.
    traded_by_class: dict[str, float] = {}
    for t in trade_rows:
        sign = -1 if t["Action"] == "SELL" else 1
        traded_by_class[t["Asset Class"]] = traded_by_class.get(t["Asset Class"], 0.0) + sign * t["Est. $"]
    # Net cash impact: + from sells, - from buys (excess parks back in Cash).
    net_buys = sum(t["Est. $"] for t in trade_rows if t["Action"] == "BUY")
    traded_by_class["Cash"] = traded_by_class.get("Cash", 0.0) + (proceeds_total - net_buys)

    post_rows = []
    for cls in classes:
        new_val = float(alloc.get(cls, 0.0)) + traded_by_class.get(cls, 0.0)
        post_rows.append({
            "Asset Class": cls,
            "Before Weight": float(alloc.get(cls, 0.0)) / broker_total if broker_total else 0,
            "After Weight": new_val / broker_total if broker_total else 0,
            "Target Weight": float(targets.get(cls, 0.0)),
        })
    post_allocation = pd.DataFrame(post_rows)
    post_allocation["In Band After"] = (
        (post_allocation["After Weight"] - post_allocation["Target Weight"]).abs()
        <= band_tolerance
    )

    stats = {
        "drift": pd.DataFrame(drift_rows),
        "proceeds_total": proceeds_total,
        "net_buys": net_buys,
        "net_cash_freed": proceeds_total - net_buys,
        "deployable_cash": deployable_cash,
        "realized_total": realized_total,
        "realized_gains": realized_gains,
        "realized_losses": realized_losses,
        "st_net": st_net,
        "lt_net": lt_net,
        "est_tax": est_tax,
        "tax_saved_vs_gains_only": tax_saved,
        "n_sells": sum(1 for t in trade_rows if t["Action"] == "SELL"),
        "n_buys": sum(1 for t in trade_rows if t["Action"] == "BUY"),
    }
    return RebalancePlan(trades=trades, post_allocation=post_allocation,
                         stats=stats, warnings=warnings)
