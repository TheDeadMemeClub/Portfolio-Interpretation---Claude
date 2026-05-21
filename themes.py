"""Theme-cluster analysis + tax-aware rebalance simulator.

Two capabilities the standard sector view misses:

1. `assign_themes` / `theme_exposure` — groups holdings into correlated
   THEMES (solar, crypto miners, materials, etc.) rather than GICS sectors.
   Ten solar stocks look diversified by sector but are really one bet; this
   surfaces that hidden concentration.

2. `simulate_rebalance` — models trimming winners + harvesting losers,
   estimating the realized gain/loss, the tax bill, and the resulting theme
   exposure, so the user can see the after-tax consequence before trading.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Theme taxonomy
# ---------------------------------------------------------------------------
# Each theme maps to (a) explicit tickers and (b) keyword patterns matched
# against the holding description. Tickers win over keywords.
THEME_TICKERS: dict[str, set[str]] = {
    "Solar & Clean Energy": {
        "ARRY", "SHLS", "RUN", "SEDG", "CSIQ", "DQ", "RNW", "BEP", "OPTT",
        "ABAT", "FSLR", "ENPH", "NOVA", "SPWR", "MAXN",
    },
    "Crypto & Miners": {
        "RIOT", "CLSK", "WULF", "MARA", "BITC", "BITQ", "COIN", "MSTR",
        "HUT", "BITF", "CIFR", "IREN", "BTBT",
    },
    "Materials & Mining": {
        "MT", "RIO", "CLF", "SID", "MP", "DOW", "PALL", "PPLT", "VALE",
        "FCX", "AA", "X", "NUE", "SCCO",
    },
    "Energy & Pipelines": {
        "APA", "EPD", "ET", "MPLX", "OXY", "DVN", "XOM", "CVX", "KMI", "WMB",
    },
    "Biotech & Pharma": {
        "QURE", "MIRA", "MRNA", "VKTX", "CRSP", "NTLA", "BEAM", "EDIT", "SAVA",
    },
    "Space & Defense": {
        "LUNR", "RKLB", "ASTS", "PL", "RDW", "SPCE",
    },
    "Semiconductors & Tech": {
        "AMD", "NVDA", "INTC", "MU", "TSM", "AVGO", "QCOM", "ARM",
    },
    "Consumer & Retail": {
        "DG", "DLTR", "WMT", "TGT", "COST", "AMZN",
    },
    "Broad Index Funds": {
        "PEOPX", "PESPX", "NASDX", "PXINX", "SCHR", "VOO", "SPY", "QQQ",
        "VTI", "IVV", "VXUS",
    },
    "Cash & Money Market": {
        "STGXX", "SPAXX", "SWVXX", "VMFXX",
    },
}

# Keyword patterns (uppercased description) -> theme, used as fallback
THEME_KEYWORDS: list[tuple[str, str]] = [
    (r"SOLAR|RENEWABLE|CLEAN ENERGY|PHOTOVOLTAIC", "Solar & Clean Energy"),
    (r"BITCOIN|CRYPTO|BLOCKCHAIN|DIGITAL ASSET|MINING", "Crypto & Miners"),
    (r"STEEL|MINING|METALS|PLATINUM|PALLADIUM|MATERIALS|COPPER|IRON", "Materials & Mining"),
    (r"ENERGY|PETROLEUM|PIPELINE|OIL|GAS|MIDSTREAM", "Energy & Pipelines"),
    (r"PHARMA|BIOTECH|THERAPEUTIC|HEALTHCARE|MEDICAL", "Biotech & Pharma"),
    (r"SPACE|AEROSPACE|DEFENSE|SATELLITE", "Space & Defense"),
    (r"SEMICONDUCTOR|MICRO DEVICES|CHIP", "Semiconductors & Tech"),
    (r"INDEX|S&P 500|NASDAQ 100|TOTAL MARKET|500 INDEX", "Broad Index Funds"),
    (r"MONEY MKT|MONEY MARKET|TREASURY BILL|CASH", "Cash & Money Market"),
    (r"NOTES|BOND|CALLABLE|CPN|DEBENTURE|FIXED RATE", "Bonds & Fixed Income"),
]

OTHER_THEME = "Other / Uncategorized"


def assign_theme(symbol: str, description: str = "", asset_type: str = "") -> str:
    """Assign a single holding to its dominant theme."""
    sym = str(symbol or "").strip().upper()
    desc = str(description or "").upper()

    # 1. Explicit ticker match
    for theme, tickers in THEME_TICKERS.items():
        if sym in tickers:
            return theme

    # 2. Bonds via asset type
    if "FIXED INCOME" in str(asset_type).upper() or "BOND" in str(asset_type).upper():
        return "Bonds & Fixed Income"

    # 3. Keyword match on description
    for pattern, theme in THEME_KEYWORDS:
        if re.search(pattern, desc):
            return theme

    return OTHER_THEME


def assign_themes(summary: pd.DataFrame) -> pd.DataFrame:
    """Add a 'Theme' column to the holdings summary."""
    if summary.empty:
        return summary.assign(Theme=pd.Series(dtype=object))
    df = summary.copy()
    df["Theme"] = df.apply(
        lambda r: assign_theme(
            r.get("Symbol", ""),
            r.get("Description", ""),
            r.get("Asset Type", ""),
        ),
        axis=1,
    )
    return df


def theme_exposure(summary_with_themes: pd.DataFrame, total_portfolio: float) -> pd.DataFrame:
    """Aggregate market value + weight by theme.

    `total_portfolio` should be the broker's authoritative total (incl. cash)
    so the weights reflect true portfolio exposure.
    """
    if summary_with_themes.empty or "Theme" not in summary_with_themes.columns:
        return pd.DataFrame()
    grp = summary_with_themes.groupby("Theme", as_index=False).agg(
        **{
            "Market Value": ("Market Value", "sum"),
            "Positions": ("Symbol", "nunique"),
            "Unrealized P&L $": ("Unrealized P&L $", "sum"),
        }
    )
    denom = total_portfolio if total_portfolio and total_portfolio > 0 else grp["Market Value"].sum()
    grp["Weight"] = grp["Market Value"] / denom if denom else np.nan
    return grp.sort_values("Market Value", ascending=False).reset_index(drop=True)


def theme_concentration_hhi(theme_df: pd.DataFrame) -> dict:
    """HHI computed on THEMES (not sectors) — the true concentration signal."""
    if theme_df.empty or "Weight" not in theme_df.columns:
        return {}
    # Exclude cash + money market from the "bet" concentration calc
    risk = theme_df[~theme_df["Theme"].isin(["Cash & Money Market", "Bonds & Fixed Income"])].copy()
    if risk.empty:
        return {}
    w = risk["Weight"] / risk["Weight"].sum()  # renormalize across risk assets
    hhi = float((w ** 2).sum())
    return {
        "hhi": hhi,
        "effective_themes": (1 / hhi) if hhi else np.nan,
        "largest_theme": risk.iloc[0]["Theme"],
        "largest_theme_weight": float(risk.iloc[0]["Weight"]),
        "n_risk_themes": int(len(risk)),
    }


# ---------------------------------------------------------------------------
# Rebalance simulator
# ---------------------------------------------------------------------------
@dataclass
class RebalanceAction:
    symbol: str
    action: str          # "Trim" or "Harvest"
    pct_of_position: float   # 0-1, how much of the position to sell
    market_value: float      # current MV of the position
    unrealized_pl: float     # current unrealized P&L $
    theme: str = ""
    is_long_term: bool = True


@dataclass
class RebalanceResult:
    actions: list[dict] = field(default_factory=list)
    total_proceeds: float = 0.0
    realized_gains: float = 0.0
    realized_losses: float = 0.0
    net_realized: float = 0.0
    lt_gains: float = 0.0
    st_gains: float = 0.0
    estimated_tax: float = 0.0
    cash_freed: float = 0.0
    theme_before: pd.DataFrame | None = None
    theme_after: pd.DataFrame | None = None


def simulate_rebalance(
    summary_with_themes: pd.DataFrame,
    actions: list[RebalanceAction],
    *,
    total_portfolio: float,
    lt_cap_gains_rate: float = 0.15,
    st_cap_gains_rate: float = 0.24,
) -> RebalanceResult:
    """Model a set of trim/harvest actions and their tax + exposure impact.

    Each action sells `pct_of_position` of the named holding. We estimate the
    realized P&L pro-rata (assuming the unrealized P&L is spread evenly across
    the position — a reasonable approximation without lot-level selection),
    then compute the tax bill and the resulting theme exposure.
    """
    result = RebalanceResult()
    df = summary_with_themes.copy()
    sym_idx = {s: i for i, s in enumerate(df["Symbol"])}

    # Track post-trade market values for the "after" exposure
    df["_post_mv"] = df["Market Value"].astype(float)

    for act in actions:
        if act.symbol not in sym_idx:
            continue
        frac = max(0.0, min(1.0, act.pct_of_position))
        proceeds = act.market_value * frac
        realized = act.unrealized_pl * frac  # pro-rata realized P&L

        result.total_proceeds += proceeds
        result.cash_freed += proceeds
        if realized >= 0:
            result.realized_gains += realized
            if act.is_long_term:
                result.lt_gains += realized
            else:
                result.st_gains += realized
        else:
            result.realized_losses += realized  # negative

        # Reduce the post-trade MV
        i = sym_idx[act.symbol]
        df.at[df.index[i], "_post_mv"] = df.at[df.index[i], "Market Value"] * (1 - frac)

        result.actions.append({
            "Symbol": act.symbol,
            "Action": act.action,
            "Theme": act.theme,
            "% Sold": frac,
            "Proceeds": round(proceeds, 2),
            "Realized P&L": round(realized, 2),
            "Term": "Long-term" if act.is_long_term else "Short-term",
        })

    result.net_realized = result.realized_gains + result.realized_losses

    # Tax: losses offset gains; net LT taxed at LT rate, net ST at ST rate.
    # Apply losses against ST first (higher rate) then LT for max benefit.
    losses = abs(result.realized_losses)
    st_taxable = result.st_gains
    lt_taxable = result.lt_gains
    used = min(losses, st_taxable)
    st_taxable -= used
    losses -= used
    used = min(losses, lt_taxable)
    lt_taxable -= used
    losses -= used

    result.estimated_tax = max(0.0, st_taxable * st_cap_gains_rate + lt_taxable * lt_cap_gains_rate)
    # Remaining losses (if any) are a deductible carryforward — informational
    result._carryforward_loss = -losses if losses > 0 else 0.0  # type: ignore[attr-defined]

    # Theme exposure before vs after
    result.theme_before = theme_exposure(df, total_portfolio)
    after = df.copy()
    after["Market Value"] = after["_post_mv"]
    # cash grows by the freed proceeds; reflect in total
    new_total = total_portfolio  # total unchanged (cash replaces securities)
    result.theme_after = theme_exposure(after, new_total)

    return result


def suggest_rebalance_actions(
    summary_with_themes: pd.DataFrame,
    *,
    trim_gain_threshold: float = 0.50,
    trim_pct: float = 0.30,
    harvest_loss_dollars: float = 250.0,
) -> list[RebalanceAction]:
    """Auto-generate sensible trim/harvest actions.

    Trim: positions up more than `trim_gain_threshold` (e.g. +50%) -> sell `trim_pct`.
    Harvest: positions with losses beyond `harvest_loss_dollars` -> sell 100%.
    """
    actions: list[RebalanceAction] = []
    if summary_with_themes.empty:
        return actions
    for _, row in summary_with_themes.iterrows():
        sym = row.get("Symbol", "")
        mv = float(row.get("Market Value", 0) or 0)
        pl = float(row.get("Unrealized P&L $", 0) or 0)
        cost = float(row.get("Total Cost", 0) or 0)
        theme = row.get("Theme", "")
        if mv <= 0:
            continue
        pl_pct = (pl / cost) if cost else 0
        if pl_pct >= trim_gain_threshold and pl > 0:
            actions.append(RebalanceAction(
                symbol=sym, action="Trim", pct_of_position=trim_pct,
                market_value=mv, unrealized_pl=pl, theme=theme, is_long_term=True,
            ))
        elif pl <= -abs(harvest_loss_dollars):
            actions.append(RebalanceAction(
                symbol=sym, action="Harvest", pct_of_position=1.0,
                market_value=mv, unrealized_pl=pl, theme=theme, is_long_term=True,
            ))
    return actions
