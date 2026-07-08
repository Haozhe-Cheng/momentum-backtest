"""
Semi-Automated Momentum Signal System
=====================================
Design choice: no full automation. Runs once a month; every order
requires manual confirmation before it is sent to the IBKR paper account.

Workflow:
  1. Compute momentum signals with yfinance
     (parameters frozen: 6-month lookback / skip 1 month / hold top 5)
  2. Compare against current paper-account positions and derive the trades needed
  3. Print the rebalance plan and wait for confirmation (orders are only sent after typing "yes")
  4. Place orders in the paper account through the TWS API
  5. Log every trade to trade_log.csv

Budget constraint: the paper account holds $1,000,000, but this system
only deploys $3,000 -- matching my real capital size so that position
sizing and slippage feel realistic.

Prerequisites:
  1. TWS is running and logged into the paper account
  2. API access is enabled in TWS settings (port 7497)
  3. pip install ib_insync yfinance pandas

Run: python signal_system.py
"""

import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    # Large-cap universe as of 2017 (same as backtest v2, no hindsight bias)
    "universe": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "INTC", "CSCO", "ORCL", "IBM",
        "NVDA", "TXN", "QCOM", "ADBE", "CRM", "ACN", "PYPL", "T", "VZ", "CMCSA",
        "BRK-B", "JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "AXP", "BLK",
        "MET", "AIG", "COF", "BK", "MA", "V",
        "JNJ", "PFE", "UNH", "MRK", "ABBV", "AMGN", "MDT", "ABT", "LLY",
        "GILD", "BMY", "CVS", "DHR",
        "PG", "KO", "PEP", "WMT", "HD", "MCD", "DIS", "NKE", "SBUX", "COST",
        "PM", "MO", "MDLZ", "TGT", "LOW", "KHC", "GM", "F", "BKNG",
        "XOM", "CVX", "COP", "SLB", "HAL", "OXY", "GE", "BA", "MMM", "HON",
        "CAT", "UNP", "UPS", "FDX", "EMR", "RTX", "LMT", "DE",
        "NEE", "DUK", "SO", "EXC",
    ],
    # Parameters are FROZEN -- selected in-sample during backtest v2,
    # no further tuning allowed
    "lookback_months": 6,
    "skip_months": 1,
    "top_n": 5,
    # Capital constraint
    "budget": 3000.0,          # the system only deploys this much, mimicking my real account
    # IBKR connection
    "tws_host": "127.0.0.1",
    "tws_port": 7497,          # paper account port
    "client_id": 17,
    "log_file": "trade_log.csv",
}


# ============================================================
# Signal computation
# ============================================================
def get_target_holdings(cfg):
    """Compute the current momentum signal and return the target holdings."""
    import yfinance as yf
    print(f"Downloading monthly data for {len(cfg['universe'])} tickers...")
    end = datetime.now().strftime("%Y-%m-%d")
    data = yf.download(cfg["universe"], period="14mo", interval="1mo",
                       auto_adjust=True, progress=False)["Close"]
    data = data.dropna(axis=1, how="any")

    # Momentum = price `skip` months ago / price `lookback` months ago - 1
    # Only completed months are used; the current unfinished month is excluded
    momentum = data.iloc[-1 - cfg["skip_months"]] / data.iloc[-1 - cfg["lookback_months"]] - 1
    momentum = momentum.dropna().sort_values(ascending=False)

    top = momentum.head(cfg["top_n"])
    print(f"\nTop {cfg['top_n']} by momentum "
          f"({cfg['lookback_months']}-month lookback, skipping the most recent {cfg['skip_months']} month(s)):")
    for tk, score in top.items():
        print(f"  {tk:<6} momentum {score:+.1%}")
    return list(top.index)


# ============================================================
# IBKR interaction
# ============================================================
def connect_ib(cfg):
    from ib_insync import IB
    ib = IB()
    try:
        ib.connect(cfg["tws_host"], cfg["tws_port"], clientId=cfg["client_id"], timeout=10)
    except Exception as e:
        print(f"\nFailed to connect to TWS: {e}")
        print("Check: 1) TWS is open and logged in  2) API access is enabled  3) port is 7497")
        sys.exit(1)
    print(f"\nConnected to paper account: {ib.managedAccounts()}")
    return ib


def get_current_positions(ib, universe):
    """Return {ticker: shares}, counting only positions within this system's universe."""
    positions = {}
    for pos in ib.positions():
        symbol = pos.contract.symbol
        if symbol in universe and pos.position != 0:
            positions[symbol] = int(pos.position)
    return positions


def get_price(ib, symbol):
    """Get a quote: request IBKR delayed data first (free); if unavailable,
    fall back to the latest yfinance close.
    Quotes are only used for share sizing, so cent-level precision is not needed."""
    from ib_insync import Stock
    ib.reqMarketDataType(3)  # 3 = delayed data, no subscription required
    contract = Stock(symbol, "SMART", "USD")
    ib.qualifyContracts(contract)
    ticker = ib.reqMktData(contract, "", False, False)
    ib.sleep(3)
    price = ticker.marketPrice()
    ib.cancelMktData(contract)
    if price is not None and not np.isnan(price) and price > 0:
        return price
    # Fallback: latest yfinance close
    import yfinance as yf
    hist = yf.Ticker(symbol).history(period="5d")["Close"]
    if len(hist) == 0:
        raise RuntimeError(f"{symbol}: no quote available from IBKR or yfinance")
    print(f"  ({symbol}: using yfinance close ${hist.iloc[-1]:.2f})")
    return float(hist.iloc[-1])


def build_rebalance_plan(ib, targets, current, budget):
    """
    Build the rebalance plan: sell holdings that dropped out of the target list,
    buy names that entered it.
    Each target stock is allocated budget / top_n.
    """
    per_stock = budget / len(targets)
    plan = []

    # Sells: held but no longer in the target list
    for symbol, shares in current.items():
        if symbol not in targets:
            price = get_price(ib, symbol)
            plan.append(("SELL", symbol, shares, price))

    # Buys: in the target list but not currently held
    for symbol in targets:
        if symbol not in current:
            price = get_price(ib, symbol)
            shares = int(per_stock // price)
            if shares >= 1:
                plan.append(("BUY", symbol, shares, price))
            else:
                print(f"  WARNING: {symbol} at ${price:.2f} exceeds the per-stock budget "
                      f"${per_stock:.0f}, skipping (a real constraint of small accounts)")
    return plan


def execute_plan(ib, plan, log_file):
    """Execute after manual confirmation, then log the trades."""
    from ib_insync import Stock, LimitOrder

    print("\n" + "=" * 50)
    print("Rebalance plan:")
    for action, symbol, shares, price in plan:
        px = f" @ ~${price:.2f}" if price else ""
        print(f"  {action:<5} {symbol:<6} {shares} shares{px}")
    print("=" * 50)

    confirm = input("\nExecute in the paper account? Type yes to proceed, anything else to cancel: ").strip().lower()
    if confirm != "yes":
        print("Cancelled. No orders were placed.")
        return

    records = []
    for action, symbol, shares, ref_price in plan:
        contract = Stock(symbol, "SMART", "USD")
        ib.qualifyContracts(contract)
        # Limit orders: buy limit = reference price +2%, sell = -2%, to ensure fills.
        # (The paper account has no market data subscription, so market orders get
        # cancelled by the system; limit orders are used to approximate market orders.)
        if action == "BUY":
            lmt = round(ref_price * 1.02, 2)
        else:
            lmt = round(ref_price * 0.98, 2)
        order = LimitOrder(action, shares, lmt, tif="GTC")
        trade = ib.placeOrder(contract, order)
        ib.sleep(3)
        status = trade.orderStatus.status
        fill_price = trade.orderStatus.avgFillPrice or 0
        print(f"  {action} {symbol} x{shares} limit ${lmt}: {status}" +
              (f" filled at ${fill_price:.2f}" if fill_price else ""))
        records.append({
            "datetime": datetime.now().isoformat(timespec="seconds"),
            "action": action, "symbol": symbol, "shares": shares,
            "status": status, "fill_price": fill_price,
            "reason": "monthly momentum rebalance",
        })

    # Append to the log
    df = pd.DataFrame(records)
    try:
        old = pd.read_csv(log_file)
        df = pd.concat([old, df], ignore_index=True)
    except FileNotFoundError:
        pass
    df.to_csv(log_file, index=False)
    print(f"\nTrades logged to {log_file}")


# ============================================================
# Main
# ============================================================
def main(cfg=CONFIG):
    print("=" * 50)
    print("  Momentum Signal System - Monthly Rebalance")
    print(f"  Budget constraint: ${cfg['budget']:,.0f} (mimicking real capital size)")
    print("=" * 50)

    targets = get_target_holdings(cfg)

    ib = connect_ib(cfg)
    try:
        current = get_current_positions(ib, cfg["universe"])
        print(f"\nCurrent positions: {current if current else '(none)'}")

        plan = build_rebalance_plan(ib, targets, current, cfg["budget"])
        if not plan:
            print("\nTarget holdings match current positions. No rebalance needed this month.")
            return
        execute_plan(ib, plan, cfg["log_file"])
    finally:
        ib.disconnect()

    print("\nNext run: the first trading day of next month. Set a calendar reminder.")


if __name__ == "__main__":
    main()
