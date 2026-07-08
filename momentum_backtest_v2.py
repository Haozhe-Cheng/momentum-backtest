"""
Momentum Factor Backtest v2 -- the "Honest Version"
=============================================
Three improvements over v1:
  1. The universe is now ~90 US large caps as of END OF 2017 -- selected by
     market cap AT THE TIME, not by which stocks later performed well.
     This substantially reduces survivorship bias.
  2. In-sample / out-of-sample split: parameters are selected only on
     2018-2022, then FROZEN and validated on 2023-2026.
     The out-of-sample result is the honest estimate of the strategy's true skill.
  3. Year-by-year return table: inspect each year separately,
     including the 2022 bear market.

Remaining limitations (honest disclosure):
  - Companies delisted or acquired after 2017 (e.g., Celgene, Time Warner)
    have no yfinance data and get dropped automatically -- this still biases
    results slightly upward. Fully fixing it requires a paid survivorship-free
    database (e.g., CRSP).
  - Renamed companies use their current tickers (FB->META, UTX->RTX,
    PCLN->BKNG, etc.).

Run: python momentum_backtest_v2.py
"""

import numpy as np
import pandas as pd

# ============================================================
# Universe: ~90 largest US companies as of end-2017 (mostly S&P 100)
# Selection criterion is "2017 market cap" -- independent of post-2018 performance
# ============================================================
UNIVERSE_2017 = [
    # Tech / communications
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "INTC", "CSCO", "ORCL", "IBM",
    "NVDA", "TXN", "QCOM", "ADBE", "CRM", "ACN", "PYPL", "T", "VZ", "CMCSA",
    # Financials
    "BRK-B", "JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "AXP", "BLK",
    "MET", "AIG", "COF", "BK", "MA", "V",
    # Healthcare
    "JNJ", "PFE", "UNH", "MRK", "ABBV", "AMGN", "MDT", "ABT", "LLY",
    "GILD", "BMY", "CVS", "DHR",
    # Consumer
    "PG", "KO", "PEP", "WMT", "HD", "MCD", "DIS", "NKE", "SBUX", "COST",
    "PM", "MO", "MDLZ", "TGT", "LOW", "KHC", "GM", "F", "BKNG",
    # Industrials / energy / utilities
    "XOM", "CVX", "COP", "SLB", "HAL", "OXY", "GE", "BA", "MMM", "HON",
    "CAT", "UNP", "UPS", "FDX", "EMR", "RTX", "LMT", "DE",
    "NEE", "DUK", "SO", "EXC",
]

CONFIG = {
    "benchmark": "SPY",
    "start_date": "2017-01-01",       # start one year early to allow a full 12-month signal window by early 2018
    "split_date": "2023-01-01",       # in-sample / out-of-sample boundary
    "end_date": "2026-06-30",
    "cost_per_trade": 0.001,
    "initial_capital": 3000,
    # Parameter combinations to test in-sample -- comparison allowed ONLY on 2018-2022!
    "param_grid": [
        {"lookback_months": 12, "skip_months": 1, "top_n": 5},
        {"lookback_months": 12, "skip_months": 1, "top_n": 10},
        {"lookback_months": 6,  "skip_months": 1, "top_n": 5},
        {"lookback_months": 6,  "skip_months": 1, "top_n": 10},
        {"lookback_months": 9,  "skip_months": 1, "top_n": 10},
    ],
}


def load_prices(tickers, start, end):
    import yfinance as yf
    print(f"Downloading {len(tickers)} tickers ({start} ~ {end}) ...")
    data = yf.download(tickers, start=start, end=end, interval="1mo",
                       auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(tickers[0])
    valid = data.columns[data.isna().mean() < 0.10]
    dropped = set(data.columns) - set(valid)
    if dropped:
        print(f"Dropped for insufficient data (residual survivorship bias source): {sorted(dropped)}")
    return data[valid].dropna(how="all")


def run_backtest(prices, params, cost):
    """Returns the portfolio's monthly return Series."""
    lookback, skip, top_n = params["lookback_months"], params["skip_months"], params["top_n"]
    signal = prices.shift(skip) / prices.shift(lookback) - 1
    monthly_ret = prices.pct_change()

    rets, prev = [], set()
    dates = prices.index[lookback + 1:]
    for date in dates:
        prev_date = prices.index[prices.index.get_loc(date) - 1]
        scores = signal.loc[prev_date].dropna()
        if len(scores) < top_n:
            rets.append(0.0)
            continue
        holdings = set(scores.nlargest(top_n).index)
        gross = monthly_ret.loc[date, list(holdings)].mean()
        turnover = len(holdings - prev) / top_n
        rets.append(gross - turnover * cost * 2)
        prev = holdings
    return pd.Series(rets, index=dates)


def metrics(r):
    r = r.dropna()
    if len(r) == 0:
        return {}
    n_years = len(r) / 12
    cum = (1 + r).prod()
    cagr = cum ** (1 / n_years) - 1
    vol = r.std() * np.sqrt(12)
    sharpe = (r.mean() * 12 - 0.03) / vol if vol > 0 else np.nan
    eq = (1 + r).cumprod()
    max_dd = (eq / eq.cummax() - 1).min()
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": max_dd,
            "WinRate": (r > 0).mean(), "CumMultiple": cum}


def print_metrics(m, label):
    print(f"  {label:<28} CAGR {m['CAGR']:>7.1%} | Sharpe {m['Sharpe']:>5.2f} | "
          f"MaxDD {m['MaxDD']:>7.1%} | Win {m['WinRate']:.0%}")


def yearly_table(port, bench):
    df = pd.DataFrame({"Strategy": port, "SPY": bench})
    yearly = df.groupby(df.index.year).apply(lambda x: (1 + x).prod() - 1)
    print(f"\n{'Year':<6}{'Strategy':>10}{'SPY':>10}{'Excess':>10}")
    for year, row in yearly.iterrows():
        print(f"{year:<6}{row['Strategy']:>10.1%}{row['SPY']:>10.1%}{row['Strategy']-row['SPY']:>10.1%}")


def main(cfg=CONFIG):
    prices = load_prices(UNIVERSE_2017, cfg["start_date"], cfg["end_date"])
    bench = load_prices([cfg["benchmark"]], cfg["start_date"], cfg["end_date"]).iloc[:, 0]
    bench_ret = bench.pct_change()

    split = pd.Timestamp(cfg["split_date"])

    # ---------- Phase 1: in-sample (2018-2022) parameter selection ----------
    print("\n" + "=" * 60)
    print("  Phase 1: In-sample parameter comparison (2018 ~ 2022)")
    print("  Rule: parameters may only be compared on this window; frozen afterwards")
    print("=" * 60)
    results = []
    for p in cfg["param_grid"]:
        r = run_backtest(prices, p, cfg["cost_per_trade"])
        r_in = r[r.index < split]
        m = metrics(r_in)
        results.append((p, m, r))
        print_metrics(m, f"{p['lookback_months']}m lookback / top {p['top_n']}")
    m_spy_in = metrics(bench_ret[(bench_ret.index < split) & (bench_ret.index >= results[0][2].index[0])])
    print_metrics(m_spy_in, "Benchmark SPY")

    # Pick the best parameters by in-sample Sharpe
    best_p, _, best_r = max(results, key=lambda x: x[1]["Sharpe"])
    print(f"\n  >>> In-sample best: {best_p['lookback_months']}m lookback / top {best_p['top_n']} (by Sharpe)")

    # ---------- Phase 2: out-of-sample (2023-2026) validation ----------
    print("\n" + "=" * 60)
    print("  Phase 2: Out-of-sample validation (2023 ~ 2026-06) -- the real score")
    print("=" * 60)
    r_out = best_r[best_r.index >= split]
    spy_out = bench_ret[bench_ret.index >= split].reindex(r_out.index)
    m_out, m_spy_out = metrics(r_out), metrics(spy_out)
    print_metrics(m_out, "Momentum (frozen params)")
    print_metrics(m_spy_out, "Benchmark SPY")
    verdict = "PASS: still beats the benchmark out-of-sample" if m_out["Sharpe"] > m_spy_out["Sharpe"] else \
              "FAIL: does not beat the benchmark out-of-sample -- the in-sample edge was likely overfitting/luck"
    print(f"\n  Verdict: {verdict}")

    # ---------- Year-by-year returns ----------
    full = best_r
    yearly_table(full, bench_ret.reindex(full.index))

    # ---------- Equity curve ----------
    try:
        import matplotlib.pyplot as plt
        eq_p = (1 + full).cumprod()
        eq_b = (1 + bench_ret.reindex(full.index).fillna(0)).cumprod()
        fig, ax = plt.subplots(figsize=(11, 5))
        eq_p.plot(ax=ax, label=f"Momentum ({best_p['lookback_months']}m/top{best_p['top_n']})")
        eq_b.plot(ax=ax, label="SPY")
        ax.axvline(split, color="red", linestyle="--", alpha=0.6, label="In/Out-of-sample split")
        ax.set_title("Momentum v2 (Honest Version) - Growth of $1")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("momentum_v2_result.png", dpi=120)
        print("\nEquity curve saved: momentum_v2_result.png (right of the red dashed line = out-of-sample)")
    except ImportError:
        pass

    print("\nHOW TO READ THESE RESULTS:")
    print("  - Focus on Phase 2 (out-of-sample) and the 2022 row -- not the full-period total return")
    print("  - If out-of-sample Sharpe is well below in-sample, the strategy has decayed or was overfit")
    print("  - Missing delisted stocks still bias results slightly upward; discount real expectations accordingly")


if __name__ == "__main__":
    main()
