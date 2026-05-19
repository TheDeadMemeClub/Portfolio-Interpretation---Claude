# Portfolio EPIC

> Private Streamlit dashboard that turns a broker positions export into an advisor-grade portfolio analysis tool — with a built-in Smart Money / M-Block trade planner, tax-loss harvesting, benchmark comparison, and AI-ready exports.

## What this app does

Upload a Wells Fargo (or similar) `.xls` / `.xlsx` / `.csv` positions export and the app generates:

- **Headline metrics** — total value, unrealized P&L, today's change, income
- **Advisor cockpit** — concentration, bearish exposure, high-risk weight flags
- **Asset allocation** — pie + horizontal bar
- **Portfolio heat map** — treemap colored by P&L %, daily change %, weight, or risk score
- **Portfolio grade scorecard** *(NEW)* — single A+ to F grade in the header, with six weighted sub-scores
- **16 analysis tabs**:
  1. **Holdings** — consolidated table with weights, P&L, trend, risk score
  2. **Trend Ratings** — EMA structure + multi-timeframe matrix (1h / 4h / 1d / 1wk)
  3. **Valuation** — fundamentals, sector treemap, sector HHI concentration index
  4. **Risk** — HHI, weighted beta, volatility, Sharpe, max drawdown, portfolio vs benchmark, Jensen's alpha + tracking error + information ratio, drawdown chart, correlation matrix, highly-correlated pairs
  5. **Winners / Losers** — top 15 each + today's movers
  6. **Income** — annual income breakdown + yield on market value
  7. **Tax Lots** — cleaned lot-level view
  8. **Rebalance** — sandbox with buy/sell deltas vs targets
  9. **💎 Smart Money** — auto-reads of top 10 holdings + full M-Block analyzer (candlestick chart, swing detection, BOS, MB zones, trade-grade checklist, risk calculator, persistent journal)
  10. **🧾 Tax-Loss Harvesting** — flags lots with harvestable losses; separates long-term vs short-term
  11. **📰 News & Earnings** — upcoming earnings calendar + latest headlines per holding
  12. **🎯 Performance** *(NEW)* — per-position contribution to return + historical backtest with rebalancing options
  13. **🎲 Projections** *(NEW)* — Monte Carlo simulation (5000 paths, 1-30 years) + named-crisis stress tests (2008, COVID, 2022, etc.)
  14. **📞 Options** *(NEW)* — covered-call screener across all stock holdings ≥100 shares + IV vs realized volatility analysis
  15. **🔬 Factors** *(NEW)* — OLS factor exposure (value, momentum, quality, growth, size, low-vol), efficient frontier with 5,000 random portfolios, dividend growth analysis
  16. **Exports** — enriched CSVs, TradingView-format CSVs, TLH candidates, full ZIP bundle, PDF dashboard summary

## What's new vs the original

| Area | Original | This version |
|------|----------|--------------|
| Smart Money planner | Standalone file, never integrated | Fully integrated tab + auto-analysis of top holdings |
| Tax-loss harvesting | ❌ | ✅ Full tab with thresholds, LT/ST split, wash-sale warning |
| Benchmark comparison | Correlation matrix only | + Cumulative-return chart, Jensen's alpha, beta, tracking error, info ratio |
| Risk scoring | Text flags only | Composite **0-100 risk score** per position + tier (Low/Moderate/Elevated/High) |
| News + earnings | ❌ | ✅ Per-holding headlines + earnings calendar |
| Sector concentration | Treemap only | + Sector HHI + effective # of sectors |
| Correlation analysis | Heatmap only | + High-correlation pair detector |
| Performance | Sequential yfinance calls | Parallel via `ThreadPoolExecutor` (~3-5x faster) |
| PDF export | ❌ | ✅ One-page advisor PDF summary |
| Theme | Default Streamlit | Custom dark theme + cohesive color palette |
| Code structure | 3 files, monolithic | 7 modular files with type hints + docstrings |

## Technical rating rule (unchanged)

```
Bullish = EMA10 > EMA20 > EMA50 AND close > 200 MA
Bearish = EMA10 < EMA20 < EMA50 AND close < 200 MA
else    = Neutral
```

Multi-timeframe matrix shows hourly / 4-hour / daily / weekly side-by-side.

## File structure

```
portfolio-epic/
├── app.py                 # Main Streamlit app (orchestrator)
├── parser.py              # Broker file parser
├── market_data.py         # Parallel yfinance fetching + news + earnings
├── analytics.py           # NEW: risk scores, TLH, benchmark, sector HHI, correlation clusters
├── smart_money.py         # NEW INTEGRATION: M-Block analyzer with persistent journal
├── exports.py             # NEW: PDF + ZIP exports
├── ui_helpers.py          # NEW: theme, formatters, components
├── .streamlit/
│   └── config.toml        # Dark theme config
├── requirements.txt
├── README.md
└── .gitignore             # Protects broker exports from being committed
```

## Privacy

Never commit broker files, exports, or CSVs to GitHub. The included `.gitignore` blocks all common patterns (`*.xls`, `*.xlsx`, `*.csv`, `WFA_*`, `portfolio_*`, etc.). The app starts empty and only analyzes the file you upload in the sidebar — nothing is persisted server-side.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Point it at `app.py`
4. Done — Streamlit Cloud will auto-install from `requirements.txt`

## Smart Money / M-Block methodology

The Smart Money tab follows the **Structure > Supply & Demand > Wyckoff** hierarchy:

- **Structure** is highest priority — recognize larger structural pushes via market blocks (MBs) and breaks of structure (BOS)
- **SND zones** act as fractal market blocks. Only body closes invalidate them
- **Wyckoff** is for confirmation, never the sole entry reason

Trade grading uses an 8-point weighted checklist (HTF bias, structure, POI, MB confirmation, Wyckoff, R:R, news, partial plan) scoring trades A+ through NO TRADE.

Risk sizing follows the 0.2%–0.5% per-trade rule from the journal.

## License

Private. Built for personal portfolio management.
