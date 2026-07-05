"""
performance.py  —  Step 6: Performance Monitoring Engine
=========================================================
Computes all metrics from trade history + live Alpaca data.

Metrics
-------
P&L          : total return, daily P&L, unrealised P&L
Risk         : Sharpe ratio, Sortino ratio, max drawdown, drawdown duration
Trade stats  : win rate, avg win, avg loss, profit factor, expectancy
Streaks      : current win/loss streak, longest win/loss streak
Breakdown    : per-trade log with entry/exit/return/duration
Equity curve : daily equity series for charting

Used by api.py — import and call compute_all(ticker, trades, equity_curve)
"""

import math
import statistics
from datetime import datetime, timedelta
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# CORE CALCULATORS
# ──────────────────────────────────────────────────────────────────────────────

def sharpe_ratio(daily_returns: list[float], risk_free: float = 0.0,
                 periods: int = 252) -> float:
    """Annualised Sharpe ratio. Returns 0 if insufficient data."""
    rets = [r - risk_free / periods for r in daily_returns]
    if len(rets) < 2:
        return 0.0
    std = statistics.stdev(rets)
    if std == 0:
        return 0.0
    return round(statistics.mean(rets) / std * math.sqrt(periods), 3)


def sortino_ratio(daily_returns: list[float], risk_free: float = 0.0,
                  periods: int = 252) -> float:
    """Annualised Sortino ratio (penalises only downside volatility)."""
    rets      = [r - risk_free / periods for r in daily_returns]
    downside  = [r for r in rets if r < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = statistics.stdev(downside)
    if downside_std == 0:
        return 0.0
    return round(statistics.mean(rets) / downside_std * math.sqrt(periods), 3)


def max_drawdown(equity: list[float]) -> dict:
    """
    Maximum drawdown and duration.
    Returns { pct, duration_days, peak_idx, trough_idx }
    """
    if len(equity) < 2:
        return {"pct": 0.0, "duration_days": 0, "peak_idx": 0, "trough_idx": 0}

    peak_idx    = 0
    trough_idx  = 0
    max_dd      = 0.0
    peak        = equity[0]
    peak_i      = 0

    for i, v in enumerate(equity):
        if v > peak:
            peak   = v
            peak_i = i
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd     = dd
            peak_idx   = peak_i
            trough_idx = i

    return {
        "pct"          : round(max_dd * 100, 3),
        "duration_days": trough_idx - peak_idx,
        "peak_idx"     : peak_idx,
        "trough_idx"   : trough_idx,
    }


def calmar_ratio(total_return_pct: float, max_dd_pct: float) -> float:
    """Annual return / max drawdown. Higher = better risk-adjusted."""
    if max_dd_pct == 0:
        return 0.0
    return round(total_return_pct / max_dd_pct, 3)


# ──────────────────────────────────────────────────────────────────────────────
# TRADE STATS
# ──────────────────────────────────────────────────────────────────────────────

def trade_statistics(trade_returns: list[float]) -> dict:
    """
    Full trade breakdown from a list of per-trade return percentages.

    trade_returns: e.g. [0.032, -0.015, 0.021, ...]  (as decimals, not %)
    """
    if not trade_returns:
        return {
            "n_trades"      : 0,
            "n_wins"        : 0,
            "n_losses"      : 0,
            "win_rate"      : 0.0,
            "avg_win"       : 0.0,
            "avg_loss"      : 0.0,
            "largest_win"   : 0.0,
            "largest_loss"  : 0.0,
            "profit_factor" : 0.0,
            "expectancy"    : 0.0,
            "avg_return"    : 0.0,
        }

    wins   = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r <= 0]

    gross_profit = sum(wins)   if wins   else 0.0
    gross_loss   = abs(sum(losses)) if losses else 0.0

    profit_factor = (
        round(gross_profit / gross_loss, 3) if gross_loss > 0
        else float("inf") if gross_profit > 0 else 0.0
    )

    win_rate  = len(wins) / len(trade_returns)
    avg_win   = statistics.mean(wins)   if wins   else 0.0
    avg_loss  = statistics.mean(losses) if losses else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    return {
        "n_trades"     : len(trade_returns),
        "n_wins"       : len(wins),
        "n_losses"     : len(losses),
        "win_rate"     : round(win_rate * 100, 2),
        "avg_win"      : round(avg_win * 100, 3),
        "avg_loss"     : round(avg_loss * 100, 3),
        "largest_win"  : round(max(wins,   default=0.0) * 100, 3),
        "largest_loss" : round(min(losses, default=0.0) * 100, 3),
        "profit_factor": profit_factor,
        "expectancy"   : round(expectancy * 100, 4),
        "avg_return"   : round(statistics.mean(trade_returns) * 100, 3),
    }


# ──────────────────────────────────────────────────────────────────────────────
# STREAK TRACKING
# ──────────────────────────────────────────────────────────────────────────────

def streak_stats(trade_returns: list[float]) -> dict:
    """
    Compute current and all-time win/loss streaks.

    Returns
    -------
    {
      current_streak     : int  (positive = wins, negative = losses)
      current_streak_type: str  ("win" | "loss" | "none")
      longest_win_streak : int
      longest_loss_streak: int
    }
    """
    if not trade_returns:
        return {
            "current_streak"     : 0,
            "current_streak_type": "none",
            "longest_win_streak" : 0,
            "longest_loss_streak": 0,
        }

    longest_win  = 0
    longest_loss = 0
    cur_win      = 0
    cur_loss     = 0

    for r in trade_returns:
        if r > 0:
            cur_win  += 1
            cur_loss  = 0
            longest_win = max(longest_win, cur_win)
        else:
            cur_loss += 1
            cur_win   = 0
            longest_loss = max(longest_loss, cur_loss)

    # Current streak from the end
    last = trade_returns[-1]
    if last > 0:
        cur = sum(1 for r in reversed(trade_returns) if r > 0)
        # stop at first loss
        streak = 0
        for r in reversed(trade_returns):
            if r > 0: streak += 1
            else: break
        return {
            "current_streak"     : streak,
            "current_streak_type": "win",
            "longest_win_streak" : longest_win,
            "longest_loss_streak": longest_loss,
        }
    else:
        streak = 0
        for r in reversed(trade_returns):
            if r <= 0: streak += 1
            else: break
        return {
            "current_streak"     : -streak,
            "current_streak_type": "loss",
            "longest_win_streak" : longest_win,
            "longest_loss_streak": longest_loss,
        }


# ──────────────────────────────────────────────────────────────────────────────
# LOG PARSER  — extracts trades from alpaca_bot.log
# ──────────────────────────────────────────────────────────────────────────────

def parse_trades_from_log(log_lines: list[str]) -> list[dict]:
    """
    Reconstruct trade history from bot log lines.

    Looks for patterns:
      BUY order submitted: N x TICKER  [id=...]
      SELL order submitted (take-profit|stop-loss|...): N x TICKER
      P&L or return info embedded in monitoring lines

    Returns list of dicts: {type, ticker, qty, time, reason}
    """
    trades = []
    entry  = None

    for line in log_lines:
        line = line.strip()
        ts   = line[:19] if len(line) > 19 else ""

        if "BUY order submitted" in line:
            # Extract: "BUY order submitted: 12 x GLD  [id=...]"
            try:
                part   = line.split("BUY order submitted:")[1].strip()
                qty_s, rest = part.split("x", 1)
                ticker = rest.split("[")[0].strip()
                entry  = {"type": "buy", "ticker": ticker,
                          "qty": int(qty_s.strip()), "time": ts}
            except Exception:
                pass

        elif "SELL order submitted" in line and entry:
            try:
                reason = "signal"
                for r in ["take-profit", "stop-loss", "trailing-stop", "GA signal"]:
                    if r in line:
                        reason = r
                        break
                part   = line.split("SELL order submitted")[1]
                part   = part.split("):")[1].strip() if "):" in part else part
                qty_s, rest = part.split("x", 1)
                ticker = rest.split("[")[0].strip()
                trades.append({
                    **entry,
                    "exit_time": ts,
                    "exit_reason": reason,
                    "exit_qty"   : int(qty_s.strip()),
                })
                entry = None
            except Exception:
                pass

    return trades


# ──────────────────────────────────────────────────────────────────────────────
# MAIN COMPUTE FUNCTION  — called by api.py
# ──────────────────────────────────────────────────────────────────────────────

def compute_all(
    equity_curve   : list[float],
    trade_returns  : list[float],   # per-trade returns as decimals
    log_lines      : list[str] = [],
) -> dict:
    """
    Master function — computes every metric in one call.

    Parameters
    ----------
    equity_curve  : daily portfolio value series (e.g. [100000, 100320, ...])
    trade_returns : per-trade return list        (e.g. [0.032, -0.015, ...])
    log_lines     : raw lines from bot log file  (for trade breakdown)

    Returns
    -------
    Full metrics dict ready to JSON-serialise and serve from /api/performance
    """
    # Daily returns from equity curve
    daily_rets = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i - 1]
        if prev > 0:
            daily_rets.append((equity_curve[i] - prev) / prev)

    total_return = (
        (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100
        if len(equity_curve) >= 2 else 0.0
    )

    dd      = max_drawdown(equity_curve)
    trades  = trade_statistics(trade_returns)
    streaks = streak_stats(trade_returns)
    parsed  = parse_trades_from_log(log_lines)

    return {
        # P&L
        "total_return_pct" : round(total_return, 3),
        "equity_start"     : round(equity_curve[0],  2) if equity_curve else 0,
        "equity_end"       : round(equity_curve[-1], 2) if equity_curve else 0,

        # Risk ratios
        "sharpe"           : sharpe_ratio(daily_rets),
        "sortino"          : sortino_ratio(daily_rets),
        "calmar"           : calmar_ratio(total_return, dd["pct"]),

        # Drawdown
        "max_drawdown_pct" : dd["pct"],
        "drawdown_duration": dd["duration_days"],

        # Trade stats
        **trades,

        # Streaks
        **streaks,

        # Raw series (for charts)
        "equity_curve"     : [round(v, 2) for v in equity_curve],
        "daily_returns"    : [round(r * 100, 4) for r in daily_rets],

        # Trade log breakdown
        "trade_log"        : parsed[-50:],   # last 50 trades
    }


# ──────────────────────────────────────────────────────────────────────────────
# MARKET HOURS CHECK  (drives 5-second vs 30-second refresh in dashboard)
# ──────────────────────────────────────────────────────────────────────────────

def is_market_hours() -> bool:
    """
    True if current ET time is within regular market hours Mon-Fri 9:30–16:00.
    Used by the dashboard to decide refresh frequency.
    """
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/New_York"))
    except ImportError:
        # Fallback: UTC-4 (EDT) or UTC-5 (EST) — close enough
        now = datetime.utcnow() - timedelta(hours=4)

    if now.weekday() >= 5:        # Saturday / Sunday
        return False
    market_open  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return market_open <= now <= market_close
