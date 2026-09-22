"""
alpaca_trades.py  —  Real trade counts from Alpaca's order history
===================================================================

Why this module exists
----------------------
The dashboard used to count trades by tailing the last 100 lines of each
bot's log file (`tail_file` in api.py). That undercounts badly — a bot that
logs a signal check every 10 minutes fills 100 lines in under a day, so any
trade older than that scrolls out of view and stops being counted. Worse,
the log files are gitignored and live only on the server, so a rebuild wipes
the history entirely and every bot drops back to "0 trades".

Alpaca's order history is the real record. It survives server rebuilds, it
covers every order the account has ever placed, and it is the same data the
broker would show you.

What it does
------------
  * Pages through the account's closed orders (Alpaca caps a page at 500).
  * Keeps only orders that actually filled (filled_qty > 0), which includes
    partially-filled-then-cancelled orders — those moved real money.
  * Pairs buys against sells FIFO, per symbol, to produce closed-trade
    counts, wins/losses and realised P&L.
  * Caches the result in memory and on disk, and refreshes in a background
    thread so the dashboard never blocks on a slow paging call.

Known limitation — shared symbols
---------------------------------
Trades are attributed by SYMBOL, because that is all an Alpaca order carries
today. The spot BTC bot and the BTC futures bot both trade BTC/USD on the
same account, so they cannot be told apart and will report the same number.
`shared_symbols()` reports which bots collide so the caller can say so
honestly rather than quietly double-counting.

The permanent fix is to tag orders at placement time with a client_order_id
that names the bot (e.g. "prom-<bot_id>-<timestamp>"). This module already
reads client_order_id and will prefer it once the bots start setting it —
see `_bot_id_from_client_order_id`.

Usage
-----
    from alpaca_trades import get_trade_counts, counts_for

    counts = get_trade_counts(trading_client)
    stats  = counts_for("GLD", counts)
    # {'trades': 12, 'wins': 8, 'win_rate': 66.7, 'realised_pnl': 412.55, ...}
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

CACHE_FILE = Path(__file__).resolve().with_name("trade_counts_cache.json")
CACHE_TTL  = 300     # seconds a cached result counts as fresh
PAGE_SIZE  = 500     # Alpaca's maximum page size for /v2/orders
MAX_PAGES  = 40      # safety cap — 20,000 orders is far more than we'll have

_CACHE = {"fetched_at": 0.0, "counts": {}, "orders_seen": 0, "error": None}
_LOCK        = threading.Lock()
_REFRESHING  = threading.Event()
_FILE_LOADED = False


# ──────────────────────────────────────────────────────────────────────────────
# SMALL HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def normalise_symbol(symbol) -> str:
    """
    Fold the several spellings of one instrument into a single key.

    Alpaca returns crypto as "BTC/USD" on orders but "BTCUSD" on positions,
    and our own configs use "BTC-USD" for file-safe names on Windows.
    """
    if not symbol:
        return ""
    return str(symbol).upper().replace("/", "").replace("-", "").replace(" ", "")


def _enum_value(v) -> str:
    """alpaca-py returns enums for side/status; older versions return plain strings."""
    if v is None:
        return ""
    return str(getattr(v, "value", v))


def _f(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _order_time(o):
    """Best available timestamp for an order, for sorting and paging."""
    for attr in ("submitted_at", "created_at", "filled_at", "updated_at"):
        ts = getattr(o, attr, None)
        if isinstance(ts, datetime):
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    return None


def _bot_id_from_client_order_id(coid: str) -> str | None:
    """
    Recognise an order this platform tagged with the bot that placed it.

    Nothing sets this yet — the bots don't pass client_order_id. Once they do
    (format "prom-<bot_id>-<anything>"), per-bot attribution works even for
    two bots sharing one symbol, and this module will pick it up with no
    further changes here.
    """
    if not coid or not str(coid).startswith("prom-"):
        return None
    parts = str(coid).split("-")
    return parts[1] if len(parts) >= 2 and parts[1] else None


# ──────────────────────────────────────────────────────────────────────────────
# FETCHING
# ──────────────────────────────────────────────────────────────────────────────

def _default_client():
    """Build a TradingClient from the environment when the caller hasn't got one."""
    from alpaca.trading.client import TradingClient
    key    = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_SECRET_KEY", "")
    paper  = os.getenv("TRADING_MODE", "paper").lower() != "live"
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY are not set")
    return TradingClient(key, secret, paper=paper)


def _closed_status():
    try:
        from alpaca.trading.enums import QueryOrderStatus
        return QueryOrderStatus.CLOSED
    except Exception:
        return "closed"


def fetch_filled_orders(client=None, max_pages: int = MAX_PAGES) -> list[dict]:
    """
    Page backwards through the account's closed orders and return every fill.

    Alpaca's /v2/orders returns newest-first by default and caps a page at 500,
    so we walk backwards using `until`. Order IDs are de-duplicated because
    `until` boundaries can overlap when several orders share a timestamp — that
    overlap is also what stops this from looping forever.
    """
    from alpaca.trading.requests import GetOrdersRequest

    if client is None:
        client = _default_client()

    status = _closed_status()
    seen: set[str] = set()
    records: list[dict] = []
    until = None

    for _ in range(max_pages):
        kwargs = {"status": status, "limit": PAGE_SIZE}
        if until is not None:
            kwargs["until"] = until

        page = client.get_orders(filter=GetOrdersRequest(**kwargs))
        if not page:
            break

        oldest = None
        fresh  = 0
        for o in page:
            ts = _order_time(o)
            if ts is not None and (oldest is None or ts < oldest):
                oldest = ts

            oid = str(getattr(o, "id", "") or "")
            if oid and oid in seen:
                continue
            if oid:
                seen.add(oid)
            fresh += 1

            rec = _to_record(o, ts)
            if rec:
                records.append(rec)

        # Nothing new on this page, a short page, or no usable timestamp all
        # mean we've reached the end of the history.
        if fresh == 0 or len(page) < PAGE_SIZE or oldest is None:
            break
        until = oldest

    return records


def _to_record(o, ts) -> dict | None:
    """Reduce an Alpaca order object to the handful of fields we need."""
    qty   = _f(getattr(o, "filled_qty", 0))
    price = _f(getattr(o, "filled_avg_price", 0))
    if qty <= 0 or price <= 0:
        return None      # never filled — not a trade

    side = _enum_value(getattr(o, "side", "")).lower()
    if side not in ("buy", "sell"):
        return None

    coid = getattr(o, "client_order_id", "") or ""
    return {
        "symbol": normalise_symbol(getattr(o, "symbol", "")),
        "side"  : side,
        "qty"   : qty,
        "price" : price,
        "at"    : ts,
        "bot_id": _bot_id_from_client_order_id(coid),
    }


# ──────────────────────────────────────────────────────────────────────────────
# FIFO PAIRING
# ──────────────────────────────────────────────────────────────────────────────

def _match(lots: deque, qty: float, price: float, short: bool):
    """
    Consume open lots FIFO against a closing order.

    Returns (realised_pnl, matched_qty, leftover_qty). Leftover means the
    closing order was bigger than anything we had open — either the opening
    order predates the window we fetched, or it's opening a position the
    other way.
    """
    pnl = 0.0
    matched = 0.0
    remaining = qty
    while remaining > 1e-12 and lots:
        lot_qty, lot_price = lots[0]
        take = min(lot_qty, remaining)
        pnl += (lot_price - price) * take if short else (price - lot_price) * take
        matched   += take
        remaining -= take
        if lot_qty - take <= 1e-12:
            lots.popleft()
        else:
            lots[0][0] = lot_qty - take
    return pnl, matched, remaining


def pair_trades(records: list[dict]) -> dict:
    """
    Turn a flat list of fills into per-symbol closed-trade statistics.

    A "trade" is a closing order: a sell that reduces a long, or a buy that
    covers a short. Opening orders aren't counted, which is what makes the
    number match how a trader would describe their activity.

    A closing order whose opening side isn't in our window can't be scored,
    so it counts toward `trades` but sits in `unknown` rather than skewing
    the win rate.
    """
    by_symbol: dict[str, list[dict]] = {}
    for r in records:
        if r["symbol"]:
            by_symbol.setdefault(r["symbol"], []).append(r)

    out: dict[str, dict] = {}
    epoch = datetime.min.replace(tzinfo=timezone.utc)

    for symbol, rows in by_symbol.items():
        rows.sort(key=lambda r: r["at"] or epoch)

        longs: deque = deque()     # [qty, price] lots we are long
        shorts: deque = deque()    # [qty, price] lots we are short
        trades = wins = losses = unknown = 0
        realised = 0.0
        last_at = None

        for r in rows:
            qty, price = r["qty"], r["price"]

            if r["side"] == "buy":
                pnl, matched, leftover = _match(shorts, qty, price, short=True)
                if matched > 1e-12:
                    trades += 1
                    realised += pnl
                    last_at = r["at"] or last_at
                    if leftover > 1e-12:
                        unknown += 1          # only partly closes — can't score it
                    elif pnl > 0:
                        wins += 1
                    else:
                        losses += 1
                if leftover > 1e-12:
                    longs.append([leftover, price])
            else:
                pnl, matched, leftover = _match(longs, qty, price, short=False)
                if matched > 1e-12:
                    trades += 1
                    realised += pnl
                    last_at = r["at"] or last_at
                    if leftover > 1e-12:
                        unknown += 1
                    elif pnl > 0:
                        wins += 1
                    else:
                        losses += 1
                    if leftover > 1e-12:
                        shorts.append([leftover, price])
                elif leftover > 1e-12:
                    # A sell with nothing open behind it: either the buy is
                    # older than our window, or the bot is opening a short.
                    shorts.append([leftover, price])

        scored = wins + losses
        out[symbol] = {
            "symbol"       : symbol,
            "trades"       : trades,
            "wins"         : wins,
            "losses"       : losses,
            "unknown"      : unknown,
            "win_rate"     : round(wins / scored * 100, 1) if scored else 0.0,
            "realised_pnl" : round(realised, 2),
            "orders"       : len(rows),
            "open_long"    : round(sum(l[0] for l in longs), 8),
            "open_short"   : round(sum(s[0] for s in shorts), 8),
            "last_trade_at": last_at.isoformat() if last_at else None,
        }

    return out


# ──────────────────────────────────────────────────────────────────────────────
# DISK CACHE
# ──────────────────────────────────────────────────────────────────────────────

def _load_cache_file() -> None:
    global _FILE_LOADED
    if _FILE_LOADED:
        return
    _FILE_LOADED = True
    try:
        if CACHE_FILE.exists():
            data = json.loads(CACHE_FILE.read_text())
            if isinstance(data.get("counts"), dict):
                _CACHE["counts"]      = data["counts"]
                _CACHE["fetched_at"]  = float(data.get("fetched_at", 0))
                _CACHE["orders_seen"] = int(data.get("orders_seen", 0))
                log.info("alpaca_trades: loaded %d symbols from cache file",
                         len(_CACHE["counts"]))
    except Exception as e:
        log.warning("alpaca_trades: could not read cache file: %s", e)


def _save_cache_file() -> None:
    try:
        tmp = CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "fetched_at" : _CACHE["fetched_at"],
            "orders_seen": _CACHE["orders_seen"],
            "counts"     : _CACHE["counts"],
        }, indent=2))
        tmp.replace(CACHE_FILE)
    except Exception as e:
        log.warning("alpaca_trades: could not write cache file: %s", e)


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def refresh_now(client=None) -> dict:
    """Fetch and re-pair everything. Blocks. Returns the fresh counts."""
    with _LOCK:
        try:
            records = fetch_filled_orders(client)
            counts  = pair_trades(records)
            _CACHE["counts"]      = counts
            _CACHE["orders_seen"] = len(records)
            _CACHE["fetched_at"]  = time.time()
            _CACHE["error"]       = None
            _save_cache_file()
            log.info("alpaca_trades: %d fills across %d symbols",
                     len(records), len(counts))
            return counts
        except Exception as e:
            _CACHE["error"] = str(e)
            log.error("alpaca_trades: refresh failed: %s", e)
            return _CACHE["counts"]


def _refresh_in_background(client=None) -> None:
    if _REFRESHING.is_set():
        return
    _REFRESHING.set()

    def _run():
        try:
            refresh_now(client)
        finally:
            _REFRESHING.clear()

    threading.Thread(target=_run, daemon=True).start()


def get_trade_counts(client=None, max_age: int = CACHE_TTL,
                     allow_blocking: bool = True) -> dict:
    """
    Per-symbol trade statistics, cached.

    Fresh cache is returned as-is. A stale cache is returned immediately and
    refreshed in the background, so a dashboard poll never waits on Alpaca.
    Only the very first call (empty cache, nothing on disk) blocks.
    """
    _load_cache_file()

    if _CACHE["counts"] and (time.time() - _CACHE["fetched_at"]) < max_age:
        return _CACHE["counts"]

    if _CACHE["counts"] or not allow_blocking:
        _refresh_in_background(client)
        return _CACHE["counts"]

    # Cold cache. /api/bots and /api/bots/all are both called on every
    # dashboard poll, so wait for any fetch already in flight rather than
    # paging Alpaca twice over for the same answer.
    with _LOCK:
        if _CACHE["counts"] and (time.time() - _CACHE["fetched_at"]) < max_age:
            return _CACHE["counts"]
    return refresh_now(client)


def counts_for(ticker: str, counts: dict | None = None) -> dict | None:
    """Statistics for one bot's ticker, or None if Alpaca has never traded it."""
    if counts is None:
        counts = get_trade_counts()
    return counts.get(normalise_symbol(ticker))


def shared_symbols(tickers) -> dict:
    """
    Which tickers collapse onto the same Alpaca symbol.

    Returns {symbol: [ticker, ...]} for symbols claimed by more than one bot —
    today that's the spot BTC bot and the BTC futures bot. Their counts are
    identical and neither is wrong, but they are the same trades seen twice.
    """
    groups: dict[str, list[str]] = {}
    for t in tickers:
        groups.setdefault(normalise_symbol(t), []).append(t)
    return {sym: ts for sym, ts in groups.items() if len(ts) > 1}


def cache_status() -> dict:
    """For /api/trades/summary and for debugging."""
    return {
        "fetched_at"  : _CACHE["fetched_at"],
        "age_seconds" : round(time.time() - _CACHE["fetched_at"], 1) if _CACHE["fetched_at"] else None,
        "orders_seen" : _CACHE["orders_seen"],
        "symbols"     : len(_CACHE["counts"]),
        "refreshing"  : _REFRESHING.is_set(),
        "error"       : _CACHE["error"],
        "cache_file"  : str(CACHE_FILE),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI  —  python alpaca_trades.py
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    counts = refresh_now()
    if not counts:
        print("No trades found (or Alpaca could not be reached).")
        print(json.dumps(cache_status(), indent=2))
        raise SystemExit(1)

    print(f"\n{'SYMBOL':<10} {'TRADES':>7} {'WINS':>6} {'LOSSES':>7} "
          f"{'WIN%':>7} {'REALISED P&L':>14}  LAST TRADE")
    print("-" * 78)
    for sym in sorted(counts, key=lambda s: -counts[s]["trades"]):
        c = counts[sym]
        last = (c["last_trade_at"] or "")[:10]
        print(f"{sym:<10} {c['trades']:>7} {c['wins']:>6} {c['losses']:>7} "
              f"{c['win_rate']:>6.1f}% {c['realised_pnl']:>14,.2f}  {last}")
    print("-" * 78)
    print(f"{'TOTAL':<10} {sum(c['trades'] for c in counts.values()):>7} "
          f"{sum(c['wins'] for c in counts.values()):>6} "
          f"{sum(c['losses'] for c in counts.values()):>7} "
          f"{'':>7} {sum(c['realised_pnl'] for c in counts.values()):>14,.2f}")

    unknown = sum(c["unknown"] for c in counts.values())
    if unknown:
        print(f"\n{unknown} closing order(s) had no matching open in the fetched "
              f"window — counted as trades, excluded from win rate.")
    print(f"\n{json.dumps(cache_status(), indent=2)}")
