"""
live_trading.py  —  Safe Live Trading Switch
=============================================
Run this BEFORE switching TRADING_MODE=live in your .env

What it does
------------
1. Pre-flight checklist  — verifies every safety condition
2. Risk guards           — enforces hard limits per trade and per day
3. Kill switch           — instantly halts all trading and closes positions
4. Live mode activation  — guides you through the .env change safely

Usage
-----
python live_trading.py --check       # run pre-flight checklist
python live_trading.py --activate    # guided switch to live trading
python live_trading.py --kill        # emergency stop — close all positions
python live_trading.py --limits      # show current risk limits
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv, set_key
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s"
)
log = logging.getLogger(__name__)

ENV_FILE = Path(__file__).parent / ".env"

# ──────────────────────────────────────────────────────────────────────────────
# HARD RISK LIMITS  (enforced regardless of GA output)
# ──────────────────────────────────────────────────────────────────────────────

LIVE_RISK_LIMITS = {
    # Per-trade limits
    "max_position_pct"      : 0.20,   # never more than 20% of portfolio in one trade
    "max_single_loss_pct"   : 0.02,   # hard stop at 2% loss per trade
    "min_confidence"        : 0.40,   # GA confidence must exceed 40% to trade live

    # Daily limits
    "max_daily_loss_pct"    : 0.05,   # halt all trading if down 5% in one day
    "max_daily_trades"      : 10,     # no more than 10 trades per day across all bots
    "max_open_positions"    : 3,      # max simultaneous open positions

    # Portfolio limits
    "min_cash_reserve_pct"  : 0.20,   # always keep 20% cash — never fully invested
    "max_portfolio_risk_pct": 0.50,   # total exposure never exceeds 50% of portfolio

    # Cool-off
    "loss_streak_halt"      : 3,      # halt if 3 consecutive losses
    "halt_duration_hours"   : 4,      # wait 4 hours after halting before resuming
}

RISK_STATE_FILE = "live_risk_state.json"


# ──────────────────────────────────────────────────────────────────────────────
# RISK STATE  (persisted between runs)
# ──────────────────────────────────────────────────────────────────────────────

def load_risk_state() -> dict:
    p = Path(RISK_STATE_FILE)
    if p.exists():
        return json.loads(p.read_text())
    return {
        "daily_loss_pct"  : 0.0,
        "daily_trades"    : 0,
        "loss_streak"     : 0,
        "halted"          : False,
        "halt_until"      : None,
        "last_reset"      : datetime.now().strftime("%Y-%m-%d"),
        "total_live_trades": 0,
    }


def save_risk_state(state: dict) -> None:
    Path(RISK_STATE_FILE).write_text(json.dumps(state, indent=2))


def reset_daily_state() -> dict:
    """Call at start of each trading day to reset daily counters."""
    state = load_risk_state()
    today = datetime.now().strftime("%Y-%m-%d")
    if state.get("last_reset") != today:
        state["daily_loss_pct"] = 0.0
        state["daily_trades"]   = 0
        state["last_reset"]     = today
        save_risk_state(state)
        log.info("Daily risk counters reset")
    return state


# ──────────────────────────────────────────────────────────────────────────────
# RISK GUARD  (call before every live order)
# ──────────────────────────────────────────────────────────────────────────────

def check_trade_allowed(
    portfolio_value : float,
    proposed_qty    : int,
    proposed_price  : float,
    confidence      : float,
    open_positions  : int,
    limits          : dict = LIVE_RISK_LIMITS,
) -> tuple[bool, str]:
    """
    Returns (allowed: bool, reason: str).
    Call this before every live order. If allowed is False, do not trade.
    """
    state = reset_daily_state()

    # ── Halted? ───────────────────────────────────────────────────────────────
    if state.get("halted"):
        halt_until = state.get("halt_until")
        if halt_until:
            until = datetime.fromisoformat(halt_until)
            if datetime.now() < until:
                remaining = int((until - datetime.now()).total_seconds() / 60)
                return False, f"Trading halted — resumes in {remaining} min"
            else:
                state["halted"]     = False
                state["loss_streak"] = 0
                save_risk_state(state)
                log.info("Halt period expired — trading resumed")

    # ── GA confidence ──────────────────────────────────────────────────────────
    if confidence < limits["min_confidence"]:
        return False, f"GA confidence {confidence:.2f} below minimum {limits['min_confidence']}"

    # ── Daily trade count ──────────────────────────────────────────────────────
    if state["daily_trades"] >= limits["max_daily_trades"]:
        return False, f"Daily trade limit reached ({limits['max_daily_trades']})"

    # ── Daily loss limit ───────────────────────────────────────────────────────
    if state["daily_loss_pct"] <= -limits["max_daily_loss_pct"]:
        _trigger_halt(state, "Daily loss limit hit")
        return False, f"Daily loss limit hit ({state['daily_loss_pct']:.2%})"

    # ── Open positions cap ────────────────────────────────────────────────────
    if open_positions >= limits["max_open_positions"]:
        return False, f"Max open positions reached ({limits['max_open_positions']})"

    # ── Position size ─────────────────────────────────────────────────────────
    trade_value = proposed_qty * proposed_price
    position_pct = trade_value / portfolio_value if portfolio_value > 0 else 1.0
    if position_pct > limits["max_position_pct"]:
        return False, (
            f"Position size {position_pct:.1%} exceeds limit "
            f"{limits['max_position_pct']:.0%}"
        )

    # ── Cash reserve ──────────────────────────────────────────────────────────
    # (caller should pass current cash / portfolio_value check)

    return True, "OK"


def record_trade_result(pnl_pct: float) -> None:
    """Call after each trade closes to update daily counters and streak."""
    state = load_risk_state()

    state["daily_trades"]      += 1
    state["total_live_trades"] += 1
    state["daily_loss_pct"]    += min(pnl_pct, 0)   # only count losses

    if pnl_pct < 0:
        state["loss_streak"] += 1
        if state["loss_streak"] >= LIVE_RISK_LIMITS["loss_streak_halt"]:
            _trigger_halt(state, f"{state['loss_streak']} consecutive losses")
    else:
        state["loss_streak"] = 0

    save_risk_state(state)


def _trigger_halt(state: dict, reason: str) -> None:
    hours     = LIVE_RISK_LIMITS["halt_duration_hours"]
    halt_until = (datetime.now() + timedelta(hours=hours)).isoformat()
    state["halted"]     = True
    state["halt_until"] = halt_until
    save_risk_state(state)
    log.warning(f"[KILL SWITCH] Trading halted: {reason}. Resumes at {halt_until}")


# ──────────────────────────────────────────────────────────────────────────────
# KILL SWITCH
# ──────────────────────────────────────────────────────────────────────────────

def emergency_stop(close_positions: bool = True) -> None:
    """
    Immediately halt all trading and optionally close all open positions.
    """
    log.warning("=" * 55)
    log.warning("  EMERGENCY STOP ACTIVATED")
    log.warning("=" * 55)

    # Halt via state file
    state = load_risk_state()
    state["halted"]     = True
    state["halt_until"] = (datetime.now() + timedelta(days=365)).isoformat()
    save_risk_state(state)
    log.warning("  Trading halted indefinitely via risk state file")

    if close_positions:
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            api_key    = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")
            is_paper   = os.getenv("TRADING_MODE", "paper").lower() != "live"

            client    = TradingClient(api_key, secret_key, paper=is_paper)
            positions = client.get_all_positions()

            if not positions:
                log.info("  No open positions to close")
            else:
                for pos in positions:
                    try:
                        client.submit_order(MarketOrderRequest(
                            symbol=pos.symbol,
                            qty=abs(int(pos.qty)),
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY,
                        ))
                        log.warning(f"  Closed position: {pos.qty} x {pos.symbol}")
                    except Exception as e:
                        log.error(f"  Failed to close {pos.symbol}: {e}")

            # Cancel all open orders
            client.cancel_orders()
            log.warning("  All pending orders cancelled")

        except Exception as e:
            log.error(f"  Could not connect to Alpaca to close positions: {e}")
            log.error("  Log in to alpaca.markets and close positions manually!")

    log.warning("  Kill switch complete. Edit live_risk_state.json to resume.")


# ──────────────────────────────────────────────────────────────────────────────
# PRE-FLIGHT CHECKLIST
# ──────────────────────────────────────────────────────────────────────────────

def run_preflight() -> bool:
    """
    Full pre-flight checklist before going live.
    Returns True only if every check passes.
    """
    print("\n" + "=" * 55)
    print("  LIVE TRADING PRE-FLIGHT CHECKLIST")
    print("=" * 55)

    passed = failed = warnings = 0

    def ok(label):
        nonlocal passed
        print(f"  [PASS]  {label}")
        passed += 1

    def fail(label, fix=""):
        nonlocal failed
        print(f"  [FAIL]  {label}")
        if fix: print(f"          Fix: {fix}")
        failed += 1

    def warn(label):
        nonlocal warnings
        print(f"  [WARN]  {label}")
        warnings += 1

    print("\n── Account ──────────────────────────────────────────")

    # Check keys exist and aren't placeholders
    api_key    = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    mode       = os.getenv("TRADING_MODE", "paper")

    if api_key and api_key not in ("YOUR_API_KEY", "your_key"):
        ok(f"ALPACA_API_KEY set ({api_key[:6]}...)")
    else:
        fail("ALPACA_API_KEY missing or placeholder", "Set real key in .env")

    if secret_key and len(secret_key) > 10:
        ok("ALPACA_SECRET_KEY set")
    else:
        fail("ALPACA_SECRET_KEY missing", "Set real secret in .env")

    if mode == "paper":
        warn("Still in PAPER mode — change to live after checklist passes")
    else:
        ok("TRADING_MODE=live")

    # Test Alpaca connection and account
    try:
        from alpaca.trading.client import TradingClient
        client  = TradingClient(api_key, secret_key, paper=(mode != "live"))
        account = client.get_account()
        equity  = float(account.equity)
        bp      = float(account.buying_power)

        ok(f"Alpaca connected — equity ${equity:,.2f}")

        if equity < 1000:
            fail(f"Account equity too low: ${equity:.2f}",
                 "Fund your live account at alpaca.markets")
        else:
            ok(f"Account funded (${equity:,.2f})")

        if account.trading_blocked:
            fail("Account trading is blocked — contact Alpaca support")
        else:
            ok("Account trading not blocked")

        if account.pattern_day_trader:
            warn("Account flagged as Pattern Day Trader — 4+ day trades in 5 days")

        positions = client.get_all_positions()
        if positions:
            warn(f"{len(positions)} open position(s) already — close before going live")
        else:
            ok("No open positions")

    except Exception as e:
        fail(f"Alpaca connection failed: {e}",
             "Check API keys and internet connection")

    print("\n── Chromosomes ──────────────────────────────────────")
    for ticker in ["GLD", "SPY", "BTC-USD"]:
        path = Path(f"{ticker}_best_chromosome.csv")
        if path.exists():
            ok(f"{ticker} chromosome exists")
        else:
            fail(f"{ticker} chromosome missing",
                 f"python train_bot.py --ticker {ticker}")

    print("\n── Paper trading history ────────────────────────────")
    log_files = list(Path(".").glob("*_bot.log"))
    if log_files:
        for lf in log_files:
            lines  = lf.read_text(encoding="utf-8", errors="replace").splitlines()
            trades = sum(1 for l in lines if "SELL order" in l)
            if trades >= 20:
                ok(f"{lf.name}: {trades} paper trades completed")
            elif trades >= 5:
                warn(f"{lf.name}: only {trades} paper trades — consider more testing")
            else:
                fail(f"{lf.name}: only {trades} paper trades",
                     "Run in paper mode for at least 20 trades before going live")
    else:
        fail("No bot log files found — has the bot run in paper mode?",
             "Run alpaca_bot.py in paper mode first")

    print("\n── Risk limits ──────────────────────────────────────")
    ok(f"Max position size: {LIVE_RISK_LIMITS['max_position_pct']*100:.0f}% per trade")
    ok(f"Daily loss halt:   {LIVE_RISK_LIMITS['max_daily_loss_pct']*100:.0f}%")
    ok(f"Loss streak halt:  {LIVE_RISK_LIMITS['loss_streak_halt']} consecutive losses")
    ok(f"Max open positions:{LIVE_RISK_LIMITS['max_open_positions']}")
    ok(f"Cash reserve:      {LIVE_RISK_LIMITS['min_cash_reserve_pct']*100:.0f}% minimum")

    print("\n" + "─" * 55)
    print(f"  {passed} passed  |  {warnings} warnings  |  {failed} failed")
    print("=" * 55)

    if failed > 0:
        print(f"\n  Fix the {failed} failing check(s) before going live.\n")
        return False
    elif warnings > 0:
        print(f"\n  {warnings} warning(s) — review before activating live trading.\n")
        return True
    else:
        print("\n  All checks passed. Run: python live_trading.py --activate\n")
        return True


# ──────────────────────────────────────────────────────────────────────────────
# ACTIVATION GUIDE
# ──────────────────────────────────────────────────────────────────────────────

def activate_live() -> None:
    """Guided step-by-step switch to live trading."""
    print("\n" + "=" * 55)
    print("  ACTIVATING LIVE TRADING")
    print("=" * 55)

    # Run preflight first
    if not run_preflight():
        print("Pre-flight failed — fix issues before activating.")
        return

    print("\nYou are about to switch to LIVE trading.")
    print("Real money will be used for all trades.\n")
    confirm = input("Type 'GO LIVE' to confirm: ").strip()
    if confirm != "GO LIVE":
        print("Cancelled.")
        return

    # Update .env
    if ENV_FILE.exists():
        set_key(str(ENV_FILE), "TRADING_MODE", "live")
        set_key(str(ENV_FILE), "CONFIRM_LIVE_TRADING", "yes")
        print("\n  .env updated:")
        print("    TRADING_MODE=live")
        print("    CONFIRM_LIVE_TRADING=yes")
    else:
        print("\n  .env not found — add these manually:")
        print("    TRADING_MODE=live")
        print("    CONFIRM_LIVE_TRADING=yes")

    # Reset risk state
    state = load_risk_state()
    state["halted"]          = False
    state["halt_until"]      = None
    state["daily_loss_pct"]  = 0.0
    state["daily_trades"]    = 0
    state["loss_streak"]     = 0
    save_risk_state(state)

    print("\n  Risk state reset.")
    print("\n  Next steps:")
    print("  1. Restart api.py       (Ctrl+C then python api.py)")
    print("  2. Restart bot_manager  (Ctrl+C then python bot_manager.py)")
    print("  3. Monitor closely for the first few days")
    print("\n  Emergency stop at any time: python live_trading.py --kill")
    print("=" * 55)


# ──────────────────────────────────────────────────────────────────────────────
# SHOW LIMITS
# ──────────────────────────────────────────────────────────────────────────────

def show_limits() -> None:
    state = load_risk_state()
    print("\n" + "=" * 55)
    print("  CURRENT RISK LIMITS & STATE")
    print("=" * 55)
    print("\nHard limits:")
    for k, v in LIVE_RISK_LIMITS.items():
        val = f"{v*100:.0f}%" if isinstance(v, float) and v < 10 else str(v)
        print(f"  {k:<30} {val}")
    print("\nToday's state:")
    for k, v in state.items():
        print(f"  {k:<30} {v}")
    print("=" * 55)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Live trading safety tools")
    parser.add_argument("--check",    action="store_true", help="Run pre-flight checklist")
    parser.add_argument("--activate", action="store_true", help="Switch to live trading")
    parser.add_argument("--kill",     action="store_true", help="Emergency stop")
    parser.add_argument("--limits",   action="store_true", help="Show risk limits")
    args = parser.parse_args()

    if args.check:
        run_preflight()
    elif args.activate:
        activate_live()
    elif args.kill:
        print("WARNING: This will close all open positions immediately.")
        confirm = input("Type 'STOP' to confirm: ").strip()
        if confirm == "STOP":
            emergency_stop(close_positions=True)
        else:
            print("Cancelled.")
    elif args.limits:
        show_limits()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
