"""
backtest.py  —  Backtesting Engine
====================================
Runs a trained chromosome against historical data and compares
performance to a simple buy-and-hold benchmark.

Usage
-----
python backtest.py --ticker GLD --start 2022-01-01 --end 2024-01-01
python backtest.py --ticker SPY --start 2023-01-01

API
---
from backtest import run_backtest
results = run_backtest("GLD", "2022-01-01", "2024-01-01")
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from stock_data import download_stock_data, add_indicators, preprocess_data
from genetic_algorithm import decode_chromosome, generate_signals, FEATURES

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CHROMOSOME LOADER
# ──────────────────────────────────────────────────────────────────────────────

def load_chromosome(ticker: str) -> np.ndarray | None:
    """Load best chromosome for a ticker. Returns None if not found."""
    yf_ticker   = ticker.replace("/", "-")
    chrom_paths = [
        Path(f"{yf_ticker}_best_chromosome.csv"),
        Path(f"{ticker}_best_chromosome.csv"),
    ]
    for path in chrom_paths:
        if path.exists():
            try:
                df    = pd.read_csv(path)
                chrom = df["weight"].values if "weight" in df.columns else df.iloc[:, 0].values
                return chrom
            except Exception as e:
                log.warning(f"Could not load chromosome from {path}: {e}")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def generate_bot_signals(df_scaled: pd.DataFrame,
                          chrom: np.ndarray,
                          weight_threshold: float = 0.15) -> pd.Series:
    """
    Generate buy/sell signals from chromosome weights.
    Returns a Series of 1 (buy), -1 (sell/short), 0 (hold).
    """
    try:
        signals = generate_signals(df_scaled, chrom, weight_threshold)
        return signals
    except Exception:
        # Fallback: simple weighted signal
        n_feat  = len(FEATURES)
        weights = chrom[:n_feat]   if len(chrom) >= n_feat else chrom
        thresh  = chrom[n_feat:n_feat*2] if len(chrom) >= n_feat*2 else np.full(n_feat, 0.5)

        weights = weights[:len(FEATURES)]
        thresh  = thresh[:len(FEATURES)]

        avail   = [f for f in FEATURES if f in df_scaled.columns]
        w       = weights[:len(avail)]
        t       = thresh[:len(avail)]

        vals      = df_scaled[avail].values
        condition = (vals > t).astype(float)
        w_sum     = w.sum()
        scores    = (condition * w).sum(axis=1) / max(w_sum, 1e-9)

        signals = pd.Series(0, index=df_scaled.index)
        signals[scores > 0.55] = 1
        signals[scores < 0.35] = -1
        return signals


# ──────────────────────────────────────────────────────────────────────────────
# BACKTEST SIMULATION
# ──────────────────────────────────────────────────────────────────────────────

def simulate_strategy(close: np.ndarray,
                       signals: np.ndarray,
                       stop_loss_pct: float   = 0.02,
                       take_profit_pct: float = 0.04,
                       starting_equity: float = 100_000.0) -> dict:
    """
    Simulate bot strategy on historical data.

    Parameters
    ----------
    close           : array of closing prices
    signals         : array of 1 (long), -1 (short), 0 (hold)
    stop_loss_pct   : fraction stop-loss
    take_profit_pct : fraction take-profit
    starting_equity : starting portfolio value

    Returns
    -------
    dict with equity_curve, trades, stats
    """
    n         = len(close)
    equity    = np.full(n, starting_equity)
    in_pos    = False
    entry     = 0.0
    direction = 1
    trades    = []

    for i in range(1, n):
        sig = int(signals[i])

        if not in_pos:
            # Only go LONG — never short ETFs/stocks on backtest
            if sig == 1:
                in_pos    = True
                entry     = close[i]
                direction = 1
        else:
            # Check stop/take-profit
            raw_ret = (close[i] - entry) / entry * direction
            closed  = False
            reason  = ""

            if raw_ret <= -stop_loss_pct:
                closed = True
                reason = "stop-loss"
            elif raw_ret >= take_profit_pct:
                closed = True
                reason = "take-profit"
            elif sig == -1:
                closed = True
                reason = "signal-exit"

            if closed:
                pnl_pct = raw_ret
                pnl_amt = equity[i-1] * pnl_pct
                trades.append({
                    "entry_idx"  : int(np.where(close == entry)[0][-1]) if entry in close else i-1,
                    "exit_idx"   : i,
                    "entry_price": round(float(entry), 4),
                    "exit_price" : round(float(close[i]), 4),
                    "direction"  : "long" if direction == 1 else "short",
                    "pnl_pct"   : round(float(pnl_pct) * 100, 3),
                    "pnl_amt"   : round(float(pnl_amt), 2),
                    "reason"    : reason,
                })
                equity[i] = equity[i-1] + pnl_amt
                in_pos    = False
            else:
                # Mark to market
                mtm       = (close[i] - close[i-1]) / close[i-1] * direction
                equity[i] = equity[i-1] * (1 + mtm)

        if not in_pos and i > 0:
            equity[i] = equity[i-1]  if equity[i] == starting_equity else equity[i]

    # Close any open position at end
    if in_pos:
        raw_ret = (close[-1] - entry) / entry * direction
        pnl_amt = equity[-2] * raw_ret
        trades.append({
            "entry_idx"  : -1,
            "exit_idx"   : n - 1,
            "entry_price": round(float(entry), 4),
            "exit_price" : round(float(close[-1]), 4),
            "direction"  : "long" if direction == 1 else "short",
            "pnl_pct"   : round(float(raw_ret) * 100, 3),
            "pnl_amt"   : round(float(pnl_amt), 2),
            "reason"    : "end-of-period",
        })
        equity[-1] = equity[-2] + pnl_amt

    return {"equity": equity, "trades": trades}


def simulate_buy_and_hold(close: np.ndarray,
                           starting_equity: float = 100_000.0) -> np.ndarray:
    """Simple buy-and-hold benchmark — buy on day 1, hold to end."""
    shares = starting_equity / close[0]
    return close * shares


# ──────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_stats(equity: np.ndarray,
                  trades: list,
                  dates: pd.DatetimeIndex,
                  label: str = "Strategy") -> dict:
    """Compute comprehensive backtest statistics."""
    returns   = pd.Series(equity).pct_change().dropna()
    total_ret = (equity[-1] / equity[0] - 1) * 100

    # Sharpe (annualised, assuming daily data)
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino
    neg_ret = returns[returns < 0]
    if len(neg_ret) > 0 and neg_ret.std() > 0:
        sortino = (returns.mean() / neg_ret.std()) * np.sqrt(252)
    else:
        sortino = 0.0

    # Max drawdown
    peak   = np.maximum.accumulate(equity)
    dd     = (peak - equity) / peak
    max_dd = dd.max() * 100

    # Win/loss stats
    wins   = [t for t in trades if t["pnl_amt"] > 0]
    losses = [t for t in trades if t["pnl_amt"] <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0

    avg_win  = np.mean([t["pnl_amt"] for t in wins])   if wins   else 0
    avg_loss = np.mean([t["pnl_amt"] for t in losses])  if losses else 0
    profit_factor = abs(sum(t["pnl_amt"] for t in wins) /
                    sum(t["pnl_amt"] for t in losses)) if losses else 999.0

    # Calmar ratio
    calmar = (total_ret / max_dd) if max_dd > 0 else 0

    # Consecutive wins/losses
    if trades:
        streaks    = [t["pnl_amt"] > 0 for t in trades]
        max_win_s  = max_consecutive(streaks, True)
        max_loss_s = max_consecutive(streaks, False)
    else:
        max_win_s = max_loss_s = 0

    return {
        "label"         : label,
        "total_return"  : round(float(total_ret), 2),
        "sharpe"        : round(float(sharpe), 3),
        "sortino"       : round(float(sortino), 3),
        "max_drawdown"  : round(float(max_dd), 2),
        "calmar"        : round(float(calmar), 3),
        "n_trades"      : len(trades),
        "win_rate"      : round(float(win_rate), 1),
        "avg_win"       : round(float(avg_win), 2),
        "avg_loss"      : round(float(avg_loss), 2),
        "profit_factor" : round(float(profit_factor), 2),
        "max_win_streak": int(max_win_s),
        "max_loss_streak": int(max_loss_s),
        "final_equity"  : round(float(equity[-1]), 2),
        "start_date"    : str(dates[0].date()) if len(dates) else "",
        "end_date"      : str(dates[-1].date()) if len(dates) else "",
    }


def max_consecutive(bools: list, value: bool) -> int:
    max_s = cur_s = 0
    for b in bools:
        if b == value:
            cur_s += 1
            max_s  = max(max_s, cur_s)
        else:
            cur_s = 0
    return max_s


# ──────────────────────────────────────────────────────────────────────────────
# MAIN BACKTEST RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def run_backtest(ticker: str,
                 start_date: str,
                 end_date: str   = None,
                 stop_loss_pct: float   = 0.02,
                 take_profit_pct: float = 0.04,
                 starting_equity: float = 100_000.0) -> dict:
    """
    Full backtest pipeline:
    1. Download historical data
    2. Compute indicators and scale
    3. Load chromosome and generate signals
    4. Simulate strategy + buy-and-hold
    5. Return full results dict for the API

    Returns
    -------
    {
      dates          : list of date strings
      bot_equity     : list of equity values (bot strategy)
      bnh_equity     : list of equity values (buy-and-hold)
      close_prices   : list of close prices
      bot_stats      : dict of bot statistics
      bnh_stats      : dict of buy-and-hold statistics
      trades         : list of trade dicts
      signals        : list of signal values (1, -1, 0)
      ticker         : str
      chromosome_found: bool
    }
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    yf_ticker = ticker.replace("/", "-")

    log.info(f"Backtesting {ticker} from {start_date} to {end_date}")

    # Download data
    df_raw = download_stock_data(yf_ticker, start_date, end_date)
    if df_raw is None or df_raw.empty:
        raise ValueError(f"No data for {ticker} in range {start_date}→{end_date}")

    df_ind    = add_indicators(df_raw.copy())
    df_scaled, _ = preprocess_data(df_ind)

    # Align indices
    idx       = df_scaled.index.intersection(df_raw.index)
    df_scaled = df_scaled.loc[idx]
    df_raw_al = df_raw.loc[idx]

    close   = df_raw_al["Close"].values
    dates   = df_scaled.index

    # Load chromosome
    chrom   = load_chromosome(ticker)
    chrom_found = chrom is not None

    if chrom_found:
        signals_series = generate_bot_signals(df_scaled, chrom)
        signals        = signals_series.values
    else:
        log.warning(f"No chromosome for {ticker} — using HOLD signals")
        signals = np.zeros(len(close))

    # Simulate
    bot_result = simulate_strategy(
        close, signals, stop_loss_pct, take_profit_pct, starting_equity
    )
    bnh_equity = simulate_buy_and_hold(close, starting_equity)

    # Stats
    bot_stats = compute_stats(bot_result["equity"], bot_result["trades"], dates, "Bot Strategy")
    bnh_stats = compute_stats(bnh_equity, [], dates, "Buy & Hold")
    bnh_stats["n_trades"]  = 1
    bnh_stats["win_rate"]  = 100.0 if bnh_equity[-1] > bnh_equity[0] else 0.0

    return {
        "ticker"          : ticker,
        "start_date"      : start_date,
        "end_date"        : end_date,
        "dates"           : [str(d.date()) for d in dates],
        "bot_equity"      : [round(float(v), 2) for v in bot_result["equity"]],
        "bnh_equity"      : [round(float(v), 2) for v in bnh_equity],
        "close_prices"    : [round(float(v), 4) for v in close],
        "signals"         : [int(s) for s in signals],
        "bot_stats"       : bot_stats,
        "bnh_stats"       : bnh_stats,
        "trades"          : bot_result["trades"],
        "chromosome_found": chrom_found,
        "n_rows"          : len(close),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtest a trained chromosome")
    parser.add_argument("--ticker", required=True, help="Ticker symbol e.g. GLD")
    parser.add_argument("--start",  required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",    default=None,  help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--stop",   type=float, default=0.02, help="Stop loss pct (default 0.02)")
    parser.add_argument("--tp",     type=float, default=0.04, help="Take profit pct (default 0.04)")
    args = parser.parse_args()

    results = run_backtest(args.ticker, args.start, args.end, args.stop, args.tp)
    bs = results["bot_stats"]
    bh = results["bnh_stats"]

    print(f"\n{'='*55}")
    print(f"  BACKTEST: {args.ticker}  {args.start} → {results['end_date']}")
    print(f"{'='*55}")
    print(f"  {'Metric':<22} {'Bot':>12} {'Buy&Hold':>12}")
    print(f"  {'-'*46}")
    print(f"  {'Total return':<22} {bs['total_return']:>+11.2f}% {bh['total_return']:>+11.2f}%")
    print(f"  {'Sharpe ratio':<22} {bs['sharpe']:>12.3f} {bh['sharpe']:>12.3f}")
    print(f"  {'Max drawdown':<22} {bs['max_drawdown']:>11.2f}% {bh['max_drawdown']:>11.2f}%")
    print(f"  {'Win rate':<22} {bs['win_rate']:>11.1f}% {bh['win_rate']:>11.1f}%")
    print(f"  {'Trades':<22} {bs['n_trades']:>12} {bh['n_trades']:>12}")
    print(f"  {'Final equity':<22} ${bs['final_equity']:>11,.2f} ${bh['final_equity']:>11,.2f}")
    print(f"{'='*55}\n")

    winner = "Bot" if bs["total_return"] > bh["total_return"] else "Buy & Hold"
    print(f"  Winner: {winner}")
    print(f"  Chromosome found: {results['chromosome_found']}")


if __name__ == "__main__":
    main()
