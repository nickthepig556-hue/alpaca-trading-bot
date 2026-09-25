#!/usr/bin/env python3
"""
patch_symbol_lookup.py  —  stop looking positions up by the wrong symbol

Run in /opt/prometheus with the bots STOPPED, AFTER patch_position_reading.py:

    systemctl stop prometheus-bots.service
    python3 patch_symbol_lookup.py

The bug
-------
Alpaca takes ORDERS for crypto as "BTC/USD" but stores the resulting POSITION
as "BTCUSD". get_open_position("BTC/USD") returns 404 Not Found — verified:

    BTC/USD    FAIL  APIError: Not Found
    BTCUSD     OK    qty=0.475512239

Both position readers wrapped that call in `except: return None`, so a 404 was
indistinguishable from "flat". run_futures_signal() only opens a position when
it believes there isn't one, so every crypto bot opened a fresh position at the
15% cap every ten minutes, forever, and never monitored any of them. That is
how one ETH position reached 95% of the account with no stop-loss.

The fix
-------
Stop using get_open_position. get_all_positions() returns whatever Alpaca
actually calls each symbol, so normalising both sides ("BTC/USD", "BTC-USD",
"btcusd" -> "BTCUSD") cannot miss.

Plus a safety net that does not depend on the lookup being right at all:
before opening, read live exposure for the symbol and refuse to add if it is
already at the cap. It fails closed — if the check itself errors, no trade.
Position detection has now been wrong twice; this makes the consequence
bounded either way.

Exact-match edits, backups alongside, aborts without changing anything if a
file doesn't look as expected. Safe to re-run.
"""

import ast
import shutil
import sys
import time
from pathlib import Path

HERE  = Path(__file__).resolve().parent
STAMP = time.strftime("%Y%m%d-%H%M%S")


HELPERS = '''def normalise_symbol(symbol) -> str:
    """
    Fold every spelling of one instrument into a single key.

    Alpaca takes crypto orders as "BTC/USD" but reports the position as
    "BTCUSD"; our configs also use "BTC-USD" for Windows-safe filenames.
    """
    return str(symbol or "").upper().replace("/", "").replace("-", "").replace(" ", "")


def find_position(trading_client, ticker):
    """
    Find an open position by normalised symbol, or None if genuinely flat.

    Deliberately does NOT use get_open_position(): that matches on the exact
    string, and for crypto the slashed form 404s even when the position very
    much exists. A 404 looks identical to "flat" to the caller, which is how
    this bot opened a new position on top of one it already held every ten
    minutes for eleven weeks.

    get_all_positions() returns Alpaca's own symbols, so normalising both
    sides can't miss regardless of which spelling the config uses.
    """
    want = normalise_symbol(ticker)
    for p in trading_client.get_all_positions():
        if normalise_symbol(getattr(p, "symbol", "")) == want:
            return p
    return None


'''

LOOKUP_OLD = """    try:
        pos = trading_client.get_open_position(ticker)
    except Exception:
        return None          # no such position — genuinely flat

    try:"""

LOOKUP_NEW = """    pos = find_position(trading_client, ticker)
    if pos is None:
        return None          # genuinely flat

    try:"""

FT_DEF = "def get_futures_position(trading_client: TradingClient, ticker: str) -> dict | None:"
AB_DEF = "def get_position(trading_client: TradingClient, ticker: str) -> dict | None:"


GUARD = '''def exposure_at_cap(trading_client, ticker, portfolio_value, risk_guard) -> bool:
    """
    True if this symbol is already held at (or near) the position cap.

    A last line of defence that does not rely on the position lookup being
    correct. Detection has been wrong twice now — once from int() on a
    fractional quantity, once from the slashed crypto symbol 404ing — and both
    times the result was unbounded stacking. This bounds it.

    Fails CLOSED: if the check itself errors we refuse the trade rather than
    assume there is room.
    """
    try:
        if not portfolio_value:
            return False
        pos = find_position(trading_client, ticker)
        if pos is None:
            return False
        pct = abs(float(pos.market_value)) / float(portfolio_value)
        if pct >= risk_guard.MAX_POSITION_PCT * 0.95:
            log.warning(f"[{ticker}] Already {pct:.1%} of portfolio "
                        f"(cap {risk_guard.MAX_POSITION_PCT:.0%}) — refusing to add")
            return True
        return False
    except Exception as e:
        log.error(f"[{ticker}] Exposure check failed ({e}) — refusing to trade")
        return True


'''

FT_LONG_OLD = '''            if qty > 0:
                oid = place_futures_order(trading_client, ticker, qty, "long", "signal")'''
FT_LONG_NEW = '''            if qty > 0 and not exposure_at_cap(trading_client, ticker, portfolio, risk_guard):
                oid = place_futures_order(trading_client, ticker, qty, "long", "signal")'''

FT_SHORT_OLD = '''            if qty > 0:
                oid = place_futures_order(trading_client, ticker, qty, "short", "signal")'''
FT_SHORT_NEW = '''            if qty > 0 and not exposure_at_cap(trading_client, ticker, portfolio, risk_guard):
                oid = place_futures_order(trading_client, ticker, qty, "short", "signal")'''


def apply(path: Path, edits, marker: str):
    if not path.exists():
        return f"SKIP  {path.name} not found"
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return f"SKIP  {path.name} already patched"
    for label, old, _ in edits:
        n = text.count(old)
        if n != 1:
            return (f"ABORT {path.name}: expected one match for '{label}', found {n}. "
                    f"Nothing changed.")
    new_text = text
    for _, old, new in edits:
        new_text = new_text.replace(old, new, 1)
    try:
        ast.parse(new_text)
    except SyntaxError as e:
        return f"ABORT {path.name}: syntax error after patch ({e}). Nothing changed."
    shutil.copy2(path, path.with_name(f"{path.name}.bak-{STAMP}"))
    path.write_text(new_text, encoding="utf-8")
    return f"OK    {path.name} patched  (backup: {path.name}.bak-{STAMP})"


def main():
    results = [
        apply(HERE / "futures_trading.py", [
            ("helpers + guard", FT_DEF,       HELPERS + GUARD + FT_DEF),
            ("position lookup", LOOKUP_OLD,   LOOKUP_NEW),
            ("long call site",  FT_LONG_OLD,  FT_LONG_NEW),
            ("short call site", FT_SHORT_OLD, FT_SHORT_NEW),
        ], marker="def find_position"),

        apply(HERE / "alpaca_bot.py", [
            ("helpers",        AB_DEF,     HELPERS + AB_DEF),
            ("position lookup", LOOKUP_OLD, LOOKUP_NEW),
        ], marker="def find_position"),
    ]

    print()
    for r in results:
        print("  " + r)
    print()
    if [r for r in results if r.startswith("ABORT")]:
        print("Nothing was changed in the files above — send me the message.")
        return 1
    print("Next: close any stacked positions, then")
    print("  systemctl start prometheus-bots.service")
    return 0


if __name__ == "__main__":
    sys.exit(main())
