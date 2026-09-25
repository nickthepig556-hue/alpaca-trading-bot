#!/usr/bin/env python3
"""
patch_dedupe_btc.py  —  one bot per symbol

Run in /opt/prometheus with the bots STOPPED:

    systemctl stop prometheus-bots.service
    python3 patch_dedupe_btc.py

The problem
-----------
BOT_CONFIGS is keyed by LABEL but every entry carries a "ticker", and
run_bot_process routes on the ticker:

    if ticker in FUTURES_CONFIGS:
        run_futures_bot(ticker)

"BTC/USD" and "BTC/USD_FUT" both carry ticker "BTC/USD", which is in
FUTURES_CONFIGS — so BOTH launch the same futures bot. Two processes trading
one symbol on one account, each opening its own 15% position. That is the 30%
BTC exposure against a 15% cap, and it is why ETH (a single entry) sat at
exactly 15%.

main() also merges dynamic bots with `BOT_CONFIGS[ticker] = cfg` — keyed by
ticker, while the static dict is keyed by label — so a bot_configs.json entry
silently replaces a hardcoded one with the same ticker.

What this does
--------------
  bot_configs.json  removes any entry whose ticker is BTC/USD (the spot "BTC
                    bot", which has never actually run as a spot bot).
  bot_manager.py    removes the hardcoded "BTC/USD" spot entry, keeping
                    "BTC/USD_FUT".
                    Adds a duplicate-ticker check at startup that logs loudly
                    instead of silently running two bots on one symbol.

Backups alongside. Aborts without changing anything if a file doesn't look as
expected. Safe to re-run.
"""

import ast
import json
import shutil
import sys
import time
from pathlib import Path

HERE  = Path(__file__).resolve().parent
STAMP = time.strftime("%Y%m%d-%H%M%S")

SPOT_BTC_BLOCK = '''"BTC/USD": {
    "id": "b1782517946", "name": "BTC bot", "ticker": "BTC/USD",
    "chromosome_file": "BTC-USD_best_chromosome.csv", "log_file": "BTC_bot.log",
    "ga": {}, "risk": {"min_allocation_pct":0.03,"max_allocation_pct":0.20,"weight_threshold":0.33,"stop_loss_pct":0.035,"take_profit_pct":0.07,"trailing_stop_pct":0.028},
    "market_open_delay_s": 0, "intraday_interval_s": 120, "lookback_days": 90, "start_date": "2010-01-01",
},

'''

DUPE_CHECK_OLD = """    tickers = [args.bot] if args.bot else list(BOT_CONFIGS.keys())"""

DUPE_CHECK_NEW = """    # Two configs sharing a ticker means two PROCESSES trading one symbol on
    # one account. They open separate positions, each sized to the full cap,
    # and each tries to manage a position the other opened. BTC/USD and
    # BTC/USD_FUT both carried ticker "BTC/USD" and both routed into
    # run_futures_bot(), which put BTC at 30% of the account against a 15%
    # cap. Fail loudly rather than let that happen quietly again.
    _by_ticker = {}
    for _key, _cfg in BOT_CONFIGS.items():
        _by_ticker.setdefault(_cfg.get("ticker"), []).append(_key)
    for _tkr, _keys in _by_ticker.items():
        if len(_keys) > 1:
            log.error(f"DUPLICATE TICKER {_tkr!r} configured by {_keys} — these "
                      f"will run as separate processes on the same symbol and "
                      f"will each open their own position. Remove one.")

    tickers = [args.bot] if args.bot else list(BOT_CONFIGS.keys())"""


def patch_manager(path: Path) -> str:
    if not path.exists():
        return f"SKIP  {path.name} not found"
    text = path.read_text(encoding="utf-8")
    if "DUPLICATE TICKER" in text:
        return f"SKIP  {path.name} already patched"

    for label, s in (("spot BTC block", SPOT_BTC_BLOCK), ("tickers line", DUPE_CHECK_OLD)):
        if text.count(s) != 1:
            return (f"ABORT {path.name}: expected one match for '{label}', "
                    f"found {text.count(s)}. Nothing changed.")

    new = text.replace(SPOT_BTC_BLOCK, "", 1).replace(DUPE_CHECK_OLD, DUPE_CHECK_NEW, 1)
    try:
        ast.parse(new)
    except SyntaxError as e:
        return f"ABORT {path.name}: syntax error after patch ({e}). Nothing changed."

    shutil.copy2(path, path.with_name(f"{path.name}.bak-{STAMP}"))
    path.write_text(new, encoding="utf-8")
    return f"OK    {path.name} patched  (backup: {path.name}.bak-{STAMP})"


def patch_configs(path: Path) -> str:
    if not path.exists():
        return f"SKIP  {path.name} not found"
    try:
        cfgs = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return f"ABORT {path.name}: could not parse ({e}). Nothing changed."

    doomed = [k for k, v in cfgs.items()
              if str(v.get("ticker", "")).upper().replace("-", "/") == "BTC/USD"]
    if not doomed:
        return f"SKIP  {path.name} has no BTC/USD entry"

    shutil.copy2(path, path.with_name(f"{path.name}.bak-{STAMP}"))
    for k in doomed:
        del cfgs[k]
    path.write_text(json.dumps(cfgs, indent=2), encoding="utf-8")
    return f"OK    {path.name}: removed {doomed}  (backup: {path.name}.bak-{STAMP})"


def main():
    results = [patch_configs(HERE / "bot_configs.json"),
               patch_manager(HERE / "bot_manager.py")]
    print()
    for r in results:
        print("  " + r)

    if [r for r in results if r.startswith("ABORT")]:
        print("\nNothing was changed — send me the message above.")
        return 1

    # Show the resulting roster so you can see exactly what will run.
    try:
        sys.path.insert(0, str(HERE))
        import importlib
        bm = importlib.import_module("bot_manager")
        from ticker_manager import load_all_configs
        merged = dict(bm.BOT_CONFIGS)
        for cfg in load_all_configs().values():
            if cfg.get("status") == "running" and Path(cfg["chromosome_file"]).exists():
                merged[cfg["ticker"]] = cfg
        by_ticker = {}
        for k, c in merged.items():
            by_ticker.setdefault(c.get("ticker"), []).append(k)
        print(f"\n  {len(merged)} bots will launch:\n")
        for tkr, keys in sorted(by_ticker.items(), key=lambda x: str(x[0])):
            flag = "   <-- DUPLICATE" if len(keys) > 1 else ""
            print(f"    {str(tkr):<10} {', '.join(keys)}{flag}")
        dupes = [t for t, k in by_ticker.items() if len(k) > 1]
        print("\n  " + ("No duplicate tickers." if not dupes
                        else f"STILL DUPLICATED: {dupes}"))
    except Exception as e:
        print(f"\n  (roster preview unavailable: {e})")

    print("\nNext: systemctl start prometheus-bots.service")
    return 0


if __name__ == "__main__":
    sys.exit(main())
