"""
Momentum Factor Backtest (v1)
=============================================
Strategy logic (classic academic 12-1 momentum, same methodology family
as Jegadeesh-Titman / AQR):
  1. At each month-end, compute every stock's trailing 12-month return,
     skipping the most recent month
     (skipping the last month avoids the short-term reversal effect --
     standard practice in the academic literature)
  2. Buy the top N stocks by momentum, equal-weighted
  3. Re-rank and rebalance at the next month-end
  4. Account for commissions and slippage

How to run:
  pip install yfinance pandas numpy matplotlib
  python momentum_backtest.py

Author: Haozhe (Eric) Cheng, 2026-07
"""

import numpy as np
import pandas as pd

# ============================================================
# Configuration -- to change the strategy, edit only this block
# ============================================================
CONFIG = {
    # Universe: feel free to add/remove. 20-50 names recommended;
    # with too few, ranking is meaningless
    "universe": [
        "AAPL", "MSFT", "NVDA", "AMD", "INTC", "GOOGL", "META", "AMZN",
        "TSLA", "AVGO", "QCOM", "MU", "SMCI", "DELL", "ORCL", "CRM",
        "JPM", "GS", "BAC", "V", "MA", "BRK-B",
        "XOM", "CVX", "COP",          # energy names
        "UNH", "JNJ", "LLY", "PG", "KO",
    ],
    "benchmark": "SPY",               # benchmark: S&P 500 ETF
    "start_date": "2018-01-01",       # backtest start (covers the 2018 selloff, 2020 COVID crash, 2022 bear market)
    "end_date": "2026-06-30",
    "lookback_months": 12,            # momentum lookback window
    "skip_months": 1,                 # skip the most recent N months (standard 12-1 momentum)
    "top_n": 5,                       # hold the top N names each month
    "cost_per_trade": 0.001,          # one-way trading cost of 0.1% (commission + slippage; optimistic for small accounts)
    "initial_capital": 3000,          # actual risk capital
}


# ============================================================
# Data layer
# ============================================================
def load_prices(tickers, start, end):
    """Download monthly adjusted close prices via yfinance.
    Returns a DataFrame: index = month-end dates, columns = tickers."""
    import yfinance as yf
    print(f"Downloading data for {len(tickers)} tickers ({start} ~ {end}) ...")
    data = yf.download(tickers, start=start, end=end, interval="1mo",
                       auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):          # yfinance returns a Series for a single ticker
        data = data.to_frame(tickers[0])
    # Drop tickers missing more than 10% of data (e.g., IPO'd mid-backtest)
    valid = data.columns[data.isna().mean() < 0.10]
    dropped = set(data.columns) - set(valid)
    if dropped:
        print(f"Dropped for insufficient data: {sorted(dropped)}")
    return data[valid].dropna(how="all")


# ============================================================
# Strategy layer
# ============================================================
def momentum_signal(prices, lookback, skip):
    """
    Compute 12-1 momentum: (price at t-skip / price at t-lookback) - 1
    Returns a DataFrame; each row holds every stock's momentum score for that month.
    """
    return prices.shift(skip) / prices.shift(lookback) - 1


def run_backtest(prices, bench_prices, cfg):
    """
    Core backtest loop. Returns (portfolio monthly returns Series, monthly holdings dict).
    """
    signal = momentum_signal(prices, cfg["lookback_months"], cfg["skip_months"])
    monthly_ret = prices.pct_change()

    port_returns = []
    holdings_log = {}
    prev_holdings = set()

    # Start at month lookback+1 (signal is incomplete before that)
    dates = prices.index[cfg["lookback_months"] + 1:]

    for i, date in enumerate(dates):
        # Use LAST month-end's signal to decide THIS month's holdings
        # -- avoids look-ahead bias
        prev_date_idx = prices.index.get_loc(date) - 1
        prev_date = prices.index[prev_date_idx]

        scores = signal.loc[prev_date].dropna()
        if len(scores) < cfg["top_n"]:
            port_returns.append(0.0)
            continue

        holdings = set(scores.nlargest(cfg["top_n"]).index)
        holdings_log[date.strftime("%Y-%m")] = sorted(holdings)

        # This month's portfolio return = equal-weighted average of held names
        gross = monthly_ret.loc[date, list(holdings)].mean()

        # Turnover cost: charged only on positions that changed
        # (one-way cost each for the buy and the sell)
        turnover = len(holdings - prev_holdings) / cfg["top_n"]
        cost = turnover * cfg["cost_per_trade"] * 2
        port_returns.append(gross - cost)
        prev_holdings = holdings

    port = pd.Series(port_returns, index=dates, name="Momentum")
    bench = bench_prices.pct_change().reindex(dates).fillna(0)
    bench.name = "SPY"
    return port, bench, holdings_log


# ============================================================
# Evaluation layer -- the handful of numbers institutions actually look at
# ============================================================
def evaluate(returns, label, capital):
    r = returns.dropna()
    n_years = len(r) / 12
    cum = (1 + r).prod()
    cagr = cum ** (1 / n_years) - 1
    vol = r.std() * np.sqrt(12)
    sharpe = (r.mean() * 12 - 0.03) / vol if vol > 0 else np.nan  # assumes a 3% risk-free rate

    equity = (1 + r).cumprod()
    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()

    print(f"\n[{label}]")
    print(f"  CAGR:              {cagr:>8.1%}")
    print(f"  Annualized vol:    {vol:>8.1%}")
    print(f"  Sharpe ratio:      {sharpe:>8.2f}   (>1 is decent; institutions only take >2 seriously)")
    print(f"  Max drawdown:      {max_dd:>8.1%}   (could you stomach your account shrinking this much?)")
    print(f"  Monthly win rate:  {(r > 0).mean():>8.1%}")
    print(f"  ${capital:,.0f} would have become:  ${capital * cum:>10,.0f}")
    return equity


# ============================================================
# Main
# ============================================================
def main(cfg=CONFIG):
    prices = load_prices(cfg["universe"], cfg["start_date"], cfg["end_date"])
    bench = load_prices([cfg["benchmark"]], cfg["start_date"], cfg["end_date"]).iloc[:, 0]

    port_ret, bench_ret, holdings = run_backtest(prices, bench, cfg)

    print("\n" + "=" * 55)
    print(f"  Momentum Factor Backtest Results  ({cfg['start_date']} ~ {cfg['end_date']})")
    print(f"  Strategy: 12-1 momentum, top {cfg['top_n']} each month, equal-weighted")
    print("=" * 55)
    eq_p = evaluate(port_ret, "Momentum strategy", cfg["initial_capital"])
    eq_b = evaluate(bench_ret, "Benchmark SPY (buy & hold)", cfg["initial_capital"])

    # Holdings over the last 6 months -- see what the strategy is actually buying
    print("\n[Holdings, last 6 months]")
    for month in list(holdings)[-6:]:
        print(f"  {month}: {', '.join(holdings[month])}")

    # Plot the equity curve (if matplotlib is installed)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        eq_p.plot(ax=ax, label="Momentum Strategy")
        eq_b.plot(ax=ax, label="SPY Buy & Hold")
        ax.set_title("Momentum Factor vs SPY (Growth of $1)")
        ax.set_ylabel("Cumulative Return")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("momentum_result.png", dpi=120)
        print("\nEquity curve saved: momentum_result.png")
    except ImportError:
        print("\n(matplotlib not installed, skipping the chart. pip install matplotlib to enable)")

    print("\nIMPORTANT CAVEATS:")
    print("  1. Backtest results != future returns. These are in-sample numbers with overfitting risk.")
    print("  2. Think through the logic before changing parameters -- don't tune repeatedly")
    print("     for a prettier curve (that's data mining).")
    print("  3. Before real money, run it on IBKR paper trading for 3 months.")


if __name__ == "__main__":
    main()
