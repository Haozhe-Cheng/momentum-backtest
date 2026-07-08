# Momentum Trading System

A momentum factor research and execution pipeline built in Python: from a naive backtest, to a survivorship-bias-corrected version with out-of-sample validation, to a semi-automated live signal system running on an Interactive Brokers paper account.

I built this to answer a simple question honestly: **does classic 12-1 momentum still work, and would it work for an account my size?** Most retail backtests answer "yes" because they are quietly rigged — hindsight-picked universes, in-sample parameter tuning, ignored costs. This project tries to remove those advantages one by one and see what survives.

## Project structure

| File | What it does |
|------|--------------|
| `momentum_backtest.py` | **v1 — naive baseline.** Classic 12-1 momentum (Jegadeesh-Titman style): rank stocks by trailing 12-month return skipping the most recent month, hold the top 5 equal-weighted, rebalance monthly. Includes trading costs. Its flaw: the universe was chosen with hindsight (today's well-known names), which inflates results. |
| `momentum_backtest_v2.py` | **v2 — the honest version.** Fixes v1's biggest problems: (1) universe = ~90 largest US stocks *as of end-2017*, selected by market cap at the time, not by later performance; (2) strict in-sample (2018–2022) / out-of-sample (2023–2026) split — parameters are chosen in-sample, then frozen and judged only on out-of-sample results; (3) year-by-year return table including the 2022 bear market. |
| `signal_system.py` | **Live execution.** Runs the frozen v2 parameters monthly, compares signals against current IBKR paper-account positions, prints a rebalance plan, and only places orders after manual confirmation. Deliberately capped at a $3,000 budget (my real capital size) despite the paper account holding $1M, so position sizing and constraints stay realistic. All trades are logged to CSV. Benchmarked against SPY through October 2026. |

## Methodology notes

- **Look-ahead bias:** each month's holdings are decided using only the prior month-end's signal.
- **Survivorship bias:** v2 uses a point-in-time (2017) universe. Residual bias remains — stocks delisted after 2017 have no free data and get dropped, which still flatters results slightly. A full fix requires a survivorship-free database like CRSP. This limitation is disclosed rather than hidden.
- **Overfitting control:** the parameter grid is compared only on 2018–2022. The out-of-sample window (2023–2026) is touched exactly once, with frozen parameters. If out-of-sample Sharpe collapses relative to in-sample, the "edge" was luck.
- **Costs:** 0.1% one-way per trade (commission + slippage), charged on turnover only.
- **Small-account realism:** the execution system skips any stock whose price exceeds the per-position budget — a real constraint of small accounts that most backtests ignore.

## How to run

```bash
pip install yfinance pandas numpy matplotlib
python momentum_backtest.py        # v1 baseline
python momentum_backtest_v2.py     # v2 with in/out-of-sample validation

# Live signals (requires IBKR TWS running with API enabled on port 7497):
pip install ib_insync
python signal_system.py
```

## What I learned

- The gap between a naive backtest and a bias-corrected one is large enough to flip conclusions. Survivorship bias alone can manufacture an "edge."
- Freezing parameters is psychologically hard — every bad month tempts you to re-tune, which is exactly how strategies get overfit.
- Execution has frictions research ignores: paper accounts without market-data subscriptions reject market orders (solved here with ±2% limit orders), and small budgets make some stocks simply unbuyable.

## Disclaimers

This is a personal research project, not investment advice. Backtested performance does not predict future returns. The live component runs exclusively on a paper (simulated) account. Code was developed with AI assistance; all design decisions, validation logic, and interpretation are my own.
