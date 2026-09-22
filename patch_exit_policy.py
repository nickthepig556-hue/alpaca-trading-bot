#!/usr/bin/env python3
"""
patch_exit_policy.py  —  make the live exit policy match the trained one

Run once, in /opt/prometheus:

    python3 patch_exit_policy.py

Four changes to alpaca_bot.py, all of which bring live behaviour back in line
with genetic_algorithm.simulate_trades() — the simulation the chromosomes were
actually scored against. None of them require retraining.

  1. trailing_stop_pct  0.015 -> 0.075   (the GA trails at stop_loss_pct * 1.5)
  2. take_profit_pct    0.50  -> 0.25    (the GA targets 0.25)
  3. A SELL signal on a long-only ticker now closes an open long instead of
     returning without touching it.
  4. max_hold_days is actually enforced — it was in the config but nothing
     ever read it.

Exact-match edits, backup written to alpaca_bot.py.bak-<timestamp>, aborts
without changing anything if the file doesn't look as expected. Safe to re-run.
"""

import ast
import shutil
import sys
import time
from pathlib import Path

HERE  = Path(__file__).resolve().parent
STAMP = time.strftime("%Y%m%d-%H%M%S")


# ── 1 + 2. Exit thresholds ────────────────────────────────────────────────────

RISK_OLD = """    'stop_loss_pct'       : 0.05,   # 5 % stop-loss below entry
    'take_profit_pct'     : 0.50,   # 4 % take-profit above entry
    'trailing_stop_pct'   : 0.015,  # 1.5 % trailing stop (activated after entry)"""

RISK_NEW = """    # These MUST match genetic_algorithm.simulate_trades(), which is the
    # simulation every chromosome was scored against. When they drift apart the
    # bot runs an exit policy the GA never optimised for, and the entry signal
    # stops meaning anything.
    #
    # The old 1.5% trailing stop was the damaging one: evaluated every 60s
    # against 1-minute prices, it closed winners on intraday noise as soon as
    # they were 5% up, while losers still ran the full 5% to the stop. Small
    # wins, full-size losses, by construction — no entry signal survives that.
    'stop_loss_pct'       : 0.05,   # 5 %  stop-loss below entry    (GA: 0.05)
    'take_profit_pct'     : 0.25,   # 25 % take-profit above entry  (GA: 0.25)
    'trailing_stop_pct'   : 0.075,  # 7.5 % trailing = stop_loss_pct * 1.5 (GA)"""


# ── 3. SELL signal on a long-only ticker ──────────────────────────────────────

LONGONLY_OLD = """        if not allow_short:
            log.info(f"Short signal ignored — {TICKER} is long-only")
            return"""

LONGONLY_NEW = """        if not allow_short:
            # We can't open a short here, but a SELL signal is still the strategy
            # saying "get out". This used to return without touching the
            # position, so for SPY/QQQ/GLD the sell signal was discarded
            # entirely and the only way out of a long was a stop or a target —
            # which is not how the GA scored these chromosomes.
            if position and position['side'] == 'long':
                log.info(f"{TICKER} is long-only — closing long on SELL signal")
                place_sell(trading_client, TICKER, position['qty'],
                           reason="signal-exit", side="long")
                alerter.trade_closed(TICKER, "long", position['qty'],
                                     position['entry_price'], current_price,
                                     "signal-exit")
            else:
                log.info(f"Short signal ignored — {TICKER} is long-only, nothing to close")
            return"""


# ── 4. max_hold_days enforcement ──────────────────────────────────────────────

MONITOR_INIT_OLD = """    trailing_stop = None   # disabled until trade moves in our favour
    peak_price    = entry_price
    trailing_activated = False"""

MONITOR_INIT_NEW = """    trailing_stop = None   # disabled until trade moves in our favour
    peak_price    = entry_price
    trailing_activated = False

    # max_hold_days has been in BOT_CONFIG all along but nothing read it, so a
    # position that never reached a stop or a target was held indefinitely. The
    # GA closes anything still open after max_hold_days, so without this the
    # live bot can sit in a trade the simulation would have exited months ago.
    # The clock starts when monitoring starts, so restarting the bot resets it
    # for an already-open position.
    opened_at  = time.time()
    max_hold_s = float(config.get('max_hold_days', 0) or 0) * 86400"""

MAXHOLD_OLD = """        qty = pos['qty']

        if side == 'short':"""

MAXHOLD_NEW = """        qty = pos['qty']

        if max_hold_s and (time.time() - opened_at) >= max_hold_s:
            held = (time.time() - opened_at) / 86400
            log.warning(f"[EXIT] MAX HOLD reached ({held:.0f} days) at {current_price:.2f}")
            place_sell(trading_client, ticker, qty, reason="max-hold", side=side)
            alerter.trade_closed(ticker, side, qty, entry_price, current_price, "max-hold")
            return True

        if side == 'short':"""


def apply(path: Path, edits, marker: str):
    if not path.exists():
        return f"SKIP  {path.name} not found"

    text = path.read_text(encoding="utf-8")
    if marker in text:
        return f"SKIP  {path.name} already patched"

    for label, old, _ in edits:
        n = text.count(old)
        if n != 1:
            return (f"ABORT {path.name}: expected one match for '{label}', "
                    f"found {n}. Nothing changed.")

    new_text = text
    for _, old, new in edits:
        new_text = new_text.replace(old, new, 1)

    try:
        ast.parse(new_text)
    except SyntaxError as e:
        return f"ABORT {path.name}: patched file has a syntax error ({e}). Nothing changed."

    shutil.copy2(path, path.with_name(f"{path.name}.bak-{STAMP}"))
    path.write_text(new_text, encoding="utf-8")
    return f"OK    {path.name} patched  (backup: {path.name}.bak-{STAMP})"


def main():
    result = apply(HERE / "alpaca_bot.py", [
        ("risk thresholds",   RISK_OLD,         RISK_NEW),
        ("long-only sell",    LONGONLY_OLD,     LONGONLY_NEW),
        ("monitor init",      MONITOR_INIT_OLD, MONITOR_INIT_NEW),
        ("max-hold check",    MAXHOLD_OLD,      MAXHOLD_NEW),
    ], marker="max_hold_s")

    print(f"\n  {result}\n")
    if result.startswith("ABORT"):
        print("Nothing was changed — send me the message above.")
        return 1

    print("Next:")
    print("  systemctl restart prometheus-bots.service")
    print("  journalctl -u prometheus-bots.service -f | grep -iE 'stop|target|monitoring'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
