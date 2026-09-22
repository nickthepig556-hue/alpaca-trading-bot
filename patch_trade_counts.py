#!/usr/bin/env python3
"""
patch_trade_counts.py  —  switch trade counts from log-tailing to Alpaca

Run once, in /opt/prometheus, with alpaca_trades.py already copied in:

    python3 patch_trade_counts.py

What it changes
---------------
  api.py          /api/bots and /api/bots/all now read trade counts from
                  Alpaca's order history (falling back to the old log
                  counting if Alpaca can't be reached), plus a new
                  /api/trades/summary endpoint for checking the numbers.
  dashboard.html  futures bots stop reporting a hardcoded 0 trades.

It is safe to run twice: it detects work already applied and stops.
Every edit is an exact match — if the file doesn't look the way this script
expects, it changes nothing and tells you, rather than guessing.
Backups are written next to each file as <name>.bak-<timestamp>.
"""

import ast
import shutil
import sys
import time
from pathlib import Path

HERE  = Path(__file__).resolve().parent
STAMP = time.strftime("%Y%m%d-%H%M%S")


# ── api.py ────────────────────────────────────────────────────────────────────

API_IMPORT_OLD = """from user_features import register_user_routes
from flask import send_from_directory
from futures_bot import FUTURES_CONFIGS
"""

API_IMPORT_NEW = """from user_features import register_user_routes
from flask import send_from_directory
from futures_bot import FUTURES_CONFIGS
from alpaca_trades import (get_trade_counts, counts_for, refresh_now,
                           shared_symbols, cache_status)
"""

API_BOTS_OLD = '''    bots    = load_bot_state()
    result  = []
    for b in bots:
        # Count trades and compute a quick return estimate from log
        lines    = tail_file(b.get("log_file", f"{b['ticker']}_bot.log"))
        buys     = sum(1 for l in lines if "BUY order" in l)
        sells    = sum(1 for l in lines if "SELL order" in l)
        wins     = sum(1 for l in lines if "take-profit" in l.lower())
        n_trades = sells   # closed trades = sells

        result.append({
            **b,
            "pid"     : b.get("pid"),
            "trades"  : n_trades,
            "buys"    : buys,
            "wins"    : wins,
            "win_rate": round(wins / max(n_trades, 1) * 100, 1),
        })
    return jsonify(result)'''

API_BOTS_NEW = '''    bots    = load_bot_state()
    # Trade counts come from Alpaca's order history, which is the only record
    # that survives a server rebuild. tail_file() only ever sees the last
    # LOG_TAIL_LINES lines, so it silently undercounts busy bots — it stays
    # here purely as a fallback for when Alpaca can't be reached.
    counts  = get_trade_counts(trading_client if alpaca_ok() else None)
    result  = []
    for b in bots:
        lines = tail_file(b.get("log_file", f"{b['ticker']}_bot.log"))
        buys  = sum(1 for l in lines if "BUY order" in l)
        stats = counts_for(b["ticker"], counts)

        if stats:
            n_trades, wins = stats["trades"], stats["wins"]
            win_rate       = stats["win_rate"]
            source         = "alpaca"
        else:
            n_trades = sum(1 for l in lines if "SELL order" in l)
            wins     = sum(1 for l in lines if "take-profit" in l.lower())
            win_rate = round(wins / max(n_trades, 1) * 100, 1)
            source   = "log"

        result.append({
            **b,
            "pid"          : b.get("pid"),
            "trades"       : n_trades,
            "buys"         : buys,
            "wins"         : wins,
            "win_rate"     : win_rate,
            "trades_source": source,
            "realised_pnl" : stats["realised_pnl"] if stats else None,
        })
    return jsonify(result)'''

API_ALL_OLD = '''    configs = load_all_configs()
    result  = []
    for bot_id, b in configs.items():
        lines    = tail_file(b.get("log_file", f"{b['ticker']}_bot.log"))
        sells    = sum(1 for l in lines if "SELL order" in l or "BUY TO COVER" in l)
        wins     = sum(1 for l in lines if "take-profit" in l.lower())
        result.append({
            **b,
            "trades"  : sells,
            "wins"    : wins,
            "win_rate": round(wins / max(sells, 1) * 100, 1),
        })
    return jsonify(result)'''

API_ALL_NEW = '''    configs = load_all_configs()
    # See /api/bots above: Alpaca's order history is the source of truth,
    # log tailing is only the fallback.
    counts  = get_trade_counts(trading_client if alpaca_ok() else None)
    result  = []
    for bot_id, b in configs.items():
        stats = counts_for(b["ticker"], counts)
        if stats:
            sells, wins = stats["trades"], stats["wins"]
            win_rate    = stats["win_rate"]
            source      = "alpaca"
        else:
            lines    = tail_file(b.get("log_file", f"{b['ticker']}_bot.log"))
            sells    = sum(1 for l in lines if "SELL order" in l or "BUY TO COVER" in l)
            wins     = sum(1 for l in lines if "take-profit" in l.lower())
            win_rate = round(wins / max(sells, 1) * 100, 1)
            source   = "log"

        result.append({
            **b,
            "trades"       : sells,
            "wins"         : wins,
            "win_rate"     : win_rate,
            "trades_source": source,
            "realised_pnl" : stats["realised_pnl"] if stats else None,
        })
    return jsonify(result)


# ── /api/trades/summary ──────────────────────────────────────────────────────
@app.route("/api/trades/summary")
def api_trades_summary():
    """
    Every symbol Alpaca has ever filled an order for, with closed-trade
    counts, win rate and realised P&L. ?refresh=1 forces a re-fetch.

    `shared` lists symbols that more than one bot claims — the spot BTC bot
    and the BTC futures bot trade the same symbol on the same account, so
    their counts are the same trades seen twice, not two separate sets.
    """
    client = trading_client if alpaca_ok() else None
    if request.args.get("refresh") == "1":
        counts = refresh_now(client)
    else:
        counts = get_trade_counts(client)

    tickers = [b["ticker"] for b in load_bot_state()]
    tickers += [b["ticker"] for b in load_all_configs().values()]
    try:
        tickers += list(FUTURES_CONFIGS.keys())
    except Exception:
        pass

    return jsonify({
        "counts" : counts,
        "totals" : {
            "trades"      : sum(c["trades"] for c in counts.values()),
            "wins"        : sum(c["wins"] for c in counts.values()),
            "losses"      : sum(c["losses"] for c in counts.values()),
            "realised_pnl": round(sum(c["realised_pnl"] for c in counts.values()), 2),
        },
        "shared" : shared_symbols(tickers),
        "cache"  : cache_status(),
    })'''


# ── dashboard.html ───────────────────────────────────────────────────────────

DASH_FETCH_OLD = """      api('/bots/all'),
      api('/futures/list').catch(() => []),
    ]);"""

DASH_FETCH_NEW = """      api('/bots/all'),
      api('/futures/list').catch(() => []),
      // Futures bots aren't in bot_configs.json, so their trade counts come
      // straight from Alpaca's order history rather than from /bots/all.
      api('/trades/summary').catch(() => null),
    ]);
    const tradeCounts = (summary && summary.counts) || {};
    const symKey = t => String(t || '').toUpperCase().replace(/[\\/\\- ]/g, '');"""

DASH_MAP_OLD = """    const futuresBots = (futuresList || [])
      .filter(f => f.trained)
      .map(f => ({
        id: 'futures_' + f.ticker.replace('/','_'),
        name: f.name, ticker: f.ticker, status: 'running',
        trades: 0, win_rate: 0, asset_type: 'futures',"""

DASH_MAP_NEW = """    const futuresBots = (futuresList || [])
      .filter(f => f.trained)
      .map(f => ({
        id: 'futures_' + f.ticker.replace('/','_'),
        name: f.name, ticker: f.ticker, status: 'running',
        trades:   (tradeCounts[symKey(f.ticker)] || {}).trades   || 0,
        win_rate: (tradeCounts[symKey(f.ticker)] || {}).win_rate || 0,
        asset_type: 'futures',"""

DASH_DESTRUCTURE_OLD = "    const [spotBots, futuresList] = await Promise.all(["
DASH_DESTRUCTURE_NEW = "    const [spotBots, futuresList, summary] = await Promise.all(["


def apply(path: Path, edits, marker: str, check_python: bool):
    """Apply a list of (label, old, new) edits to one file, or change nothing."""
    if not path.exists():
        return f"SKIP  {path.name} not found"

    text = path.read_text(encoding="utf-8")
    if marker in text:
        return f"SKIP  {path.name} already patched"

    for label, old, _ in edits:
        n = text.count(old)
        if n != 1:
            return (f"ABORT {path.name}: expected exactly one match for "
                    f"'{label}', found {n}. Nothing was changed.")

    new_text = text
    for _, old, new in edits:
        new_text = new_text.replace(old, new, 1)

    if check_python:
        try:
            ast.parse(new_text)
        except SyntaxError as e:
            return f"ABORT {path.name}: patched file has a syntax error ({e}). Nothing was changed."

    backup = path.with_name(f"{path.name}.bak-{STAMP}")
    shutil.copy2(path, backup)
    path.write_text(new_text, encoding="utf-8")
    return f"OK    {path.name} patched  (backup: {backup.name})"


def main():
    if not (HERE / "alpaca_trades.py").exists():
        print("ABORT alpaca_trades.py is not in this directory — copy it here first.")
        return 1

    results = [
        apply(HERE / "api.py", [
            ("import block",       API_IMPORT_OLD, API_IMPORT_NEW),
            ("/api/bots",          API_BOTS_OLD,   API_BOTS_NEW),
            ("/api/bots/all",      API_ALL_OLD,    API_ALL_NEW),
        ], marker="from alpaca_trades import", check_python=True),

        apply(HERE / "dashboard.html", [
            ("loadBots destructure", DASH_DESTRUCTURE_OLD, DASH_DESTRUCTURE_NEW),
            ("loadBots fetches",     DASH_FETCH_OLD,       DASH_FETCH_NEW),
            ("futures bot mapping",  DASH_MAP_OLD,         DASH_MAP_NEW),
        ], marker="/trades/summary", check_python=False),
    ]

    print()
    for r in results:
        print("  " + r)
    print()

    failed = [r for r in results if r.startswith("ABORT")]
    if failed:
        print("Nothing was changed in the files above. Send me the message and")
        print("I'll adjust the patch to match what's actually on the server.")
        return 1

    print("Next:")
    print("  python3 alpaca_trades.py          # check the counts Alpaca reports")
    print("  systemctl restart prometheus-api.service")
    return 0


if __name__ == "__main__":
    sys.exit(main())
