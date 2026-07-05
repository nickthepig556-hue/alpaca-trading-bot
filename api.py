"""
api.py  —  Step 5: Flask API Server
=====================================
Bridges dashboard.html to your live Alpaca paper/live account.

Endpoints
---------
GET  /api/account       → portfolio value, buying power, P&L
GET  /api/positions     → open positions with unrealised P&L
GET  /api/bots          → bot status read from bot_state.json
GET  /api/logs          → last N lines from GLD_bot.log
GET  /api/performance   → daily equity curve + Sharpe + drawdown
POST /api/bots/pause    → pause a running bot  { "id": "b1" }
POST /api/bots/resume   → resume a paused bot  { "id": "b1" }
POST /api/bots/create   → launch a new bot     { ...config }

Setup
-----
pip install flask flask-cors alpaca-py python-dotenv

Run
---
python api.py           (starts on http://localhost:5000)
"""

import os
import json
import math
import time
import logging
import threading
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
from performance import compute_all, is_market_hours, parse_trades_from_log
import threading
from pathlib import Path
from ticker_manager import (validate_ticker, create_bot_config, search_tickers, load_all_configs, add_bot_config, remove_bot_config, update_bot_status, POPULAR_TICKERS)

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify, request, g
from auth import (
    register_user, login_user, get_user_from_token,
    get_user_profile, get_global_stats, get_all_users,
    require_auth, require_admin, delete_session,
    add_user_bot, get_user_bots, record_trade
)
from flask_cors import CORS

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import threading
from pathlib import Path
from ticker_manager import (
    validate_ticker, create_bot_config, search_tickers,
    load_all_configs, add_bot_config, remove_bot_config,
    update_bot_status, POPULAR_TICKERS

)
from user_features import register_user_routes
# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

API_KEY    = os.getenv("ALPACA_API_KEY", "")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
IS_PAPER   = os.getenv("TRADING_MODE", "paper").lower() != "live"

BOT_STATE_FILE = "bot_state.json"   # persists bot on/off state across restarts
LOG_TAIL_LINES = 100                 # how many log lines to serve

# ──────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)   # allow dashboard.html (file://) to call localhost:5000

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# ALPACA CLIENTS  (created once, reused)
# ──────────────────────────────────────────────────────────────────────────────

def get_clients():
    trading = TradingClient(API_KEY, SECRET_KEY, paper=IS_PAPER)
    data    = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    return trading, data

try:
    trading_client, data_client = get_clients()
    log.info(f"Alpaca connected  [{'PAPER' if IS_PAPER else 'LIVE'}]")
except Exception as e:
    trading_client = data_client = None
    log.error(f"Alpaca connection failed: {e}")


def alpaca_ok():
    return trading_client is not None


# ──────────────────────────────────────────────────────────────────────────────
# BOT STATE  (JSON file so dashboard survives server restarts)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_BOTS = [
    {
        "id": "b1", "name": "GLD bot", "ticker": "GLD",
        "status": "running", "gens": 200, "pop": 100,
        "alloc": 40, "stop": 2.0, "pid": None,
        "chromosome_file": "GLD_best_chromosome.csv",
        "log_file": "GLD_bot.log",
    },
]

def load_bot_state() -> list:
    if Path(BOT_STATE_FILE).exists():
        with open(BOT_STATE_FILE) as f:
            return json.load(f)
    save_bot_state(DEFAULT_BOTS)
    return DEFAULT_BOTS

def save_bot_state(bots: list) -> None:
    with open(BOT_STATE_FILE, "w") as f:
        json.dump(bots, f, indent=2)

def get_bot(bot_id: str) -> dict | None:
    return next((b for b in load_bot_state() if b["id"] == bot_id), None)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def tail_file(path: str, n: int = 100) -> list[str]:
    """Return last n lines of a text file without loading the whole thing."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8", errors="replace") as f:
        return list(deque(f, maxlen=n))


def parse_log_lines(lines: list[str]) -> list[dict]:
    """Convert raw log lines into structured dicts for the dashboard."""
    entries = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        parts = line.split("  ", 2)
        if len(parts) < 3:
            entries.append({"time": "", "level": "INFO", "msg": line})
            continue
        ts, level, msg = parts[0], parts[1].strip(), parts[2]
        # Classify for dashboard colour coding
        ltype = "info"
        if "BUY" in msg or "[OK]" in msg:
            ltype = "buy"
        elif "SELL" in msg or "[STOP]" in msg or "[TARGET]" in msg:
            ltype = "sell"
        elif "ERROR" in level or "WARNING" in level or "WARN" in level:
            ltype = "warn"
        entries.append({"time": ts[11:16], "level": level, "msg": msg, "type": ltype})
    return entries[:LOG_TAIL_LINES]


# ──────────────────────────────────────────────────────────────────────────────
# PERFORMANCE METRICS
# ──────────────────────────────────────────────────────────────────────────────

def fetch_daily_bars(ticker: str, days: int = 90) -> list[float]:
    """Return list of daily close prices — crypto via Alpaca, stocks via yfinance."""
    import pandas as pd
    CRYPTO_TICKERS = {"BTC/USD", "ETH/USD", "LTC/USD", "BCH/USD"}
    try:
        if ticker in CRYPTO_TICKERS:
            from alpaca.data.historical import CryptoHistoricalDataClient
            from alpaca.data.requests import CryptoBarsRequest
            from alpaca.data.timeframe import TimeFrame
            end   = datetime.now()
            start = end - timedelta(days=days + 10)
            client = CryptoHistoricalDataClient()
            req    = CryptoBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=start.isoformat(),
                end=end.isoformat(),
            )
            bars = client.get_crypto_bars(req).df
            if bars.empty:
                return []
            if hasattr(bars.index, "levels"):
                bars = bars.xs(ticker, level="symbol")
            return bars["close"].tolist()[-days:]
        else:
            import yfinance as yf
            yf_ticker = ticker.replace("/", "-")
            df = yf.download(yf_ticker,
                             period=f"{days + 10}d",
                             auto_adjust=True,
                             progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty:
                return []
            return [float(c) for c in df["Close"].dropna().tolist()][-days:]
    except Exception as e:
        log.warning(f"fetch_daily_bars({ticker}): {e}")
        return []

    import statistics

    # Equity curve: assume fully invested
    base   = closes[0]
    equity = [round(start_value * c / base, 2) for c in closes]

    # Daily returns
    rets   = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]

    # Total return
    total_return = (closes[-1] - closes[0]) / closes[0]

    # Sharpe (annualised, assumes 252 trading days)
    if len(rets) > 1 and statistics.stdev(rets) > 0:
        sharpe = (statistics.mean(rets) / statistics.stdev(rets)) * math.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    peak, max_dd = equity[0], 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    return {
        "equity"      : equity,
        "total_return": round(total_return * 100, 2),
        "sharpe"      : round(sharpe, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "daily_returns": [round(r * 100, 4) for r in rets],
    }


# ──────────────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────────────

# ── /api/account ──────────────────────────────────────────────────────────────
@app.route("/api/account")
def api_account():
    """Live portfolio value, P&L, buying power from Alpaca."""
    if not alpaca_ok():
        return jsonify({"error": "Alpaca not connected"}), 503
    try:
        acct = trading_client.get_account()
        return jsonify({
            "portfolio_value" : round(safe_float(acct.portfolio_value), 2),
            "equity"          : round(safe_float(acct.equity), 2),
            "buying_power"    : round(safe_float(acct.buying_power), 2),
            "cash"            : round(safe_float(acct.cash), 2),
            "pnl_today"       : round(
                safe_float(acct.equity) - safe_float(acct.last_equity), 2
            ),
            "pnl_today_pct"   : round(
                (safe_float(acct.equity) - safe_float(acct.last_equity))
                / max(safe_float(acct.last_equity), 1) * 100, 3
            ),
            "status"          : str(acct.status),
            "mode"            : "paper" if IS_PAPER else "live",
        })
    except Exception as e:
        log.error(f"/api/account error: {e}")
        return jsonify({"error": str(e)}), 500


# ── /api/positions ────────────────────────────────────────────────────────────
@app.route("/api/positions")
def api_positions():
    """All open positions with live unrealised P&L."""
    if not alpaca_ok():
        return jsonify([]), 503
    try:
        positions = trading_client.get_all_positions()
        return jsonify([
            {
                "symbol"       : p.symbol,
                "qty"          : int(p.qty),
                "entry_price"  : round(safe_float(p.avg_entry_price), 2),
                "current_price": round(safe_float(p.current_price), 2),
                "market_value" : round(safe_float(p.market_value), 2),
                "unrealised_pl": round(safe_float(p.unrealized_pl), 2),
                "unrealised_pct": round(safe_float(p.unrealized_plpc) * 100, 3),
                "side"         : str(p.side),
            }
            for p in positions
        ])
    except Exception as e:
        log.error(f"/api/positions error: {e}")
        return jsonify({"error": str(e)}), 500


# ── /api/bots ─────────────────────────────────────────────────────────────────
@app.route("/api/bots")
def api_bots():
    """Bot list with status, return, and trade count pulled from log file."""
    bots    = load_bot_state()
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
    return jsonify(result)


# ── /api/logs ────────────────────────────────────────────────────────────────
@app.route("/api/logs")
def api_logs():
    """Trade log entries from the bot's log file."""
    bot_id   = request.args.get("bot_id", "")
    log_file = "GLD_bot.log"   # default fallback

    # Check bot_configs.json first (dynamic bots)
    dynamic = load_all_configs()
    if bot_id in dynamic:
        log_file = dynamic[bot_id].get("log_file", log_file)
    else:
        # Fall back to bot_state.json (legacy bots)
        bot = get_bot(bot_id)
        if bot:
            log_file = bot.get("log_file", log_file)

    lines = tail_file(log_file)
    return jsonify(parse_log_lines(lines))

# ── /api/performance ─────────────────────────────────────────────────────────
@app.route("/api/performance")
def api_performance():
    """
    Full performance metrics: P&L, Sharpe, Sortino, Calmar, drawdown,
    win rate, streaks, trade breakdown, equity curve.
    ?ticker=GLD&days=90&bot_id=b1
    """
    ticker = request.args.get("ticker", "GLD")
    days   = int(request.args.get("days", 90))
    bot_id = request.args.get("bot_id", "b1")

    closes    = fetch_daily_bars(ticker, days)
    log_lines = tail_file(get_bot(bot_id)["log_file"]) if get_bot(bot_id) else []

    # Build equity curve normalised to starting portfolio value (100k default)
    start_val = 100_000.0
    if closes and len(closes) > 1 and closes[0] and closes[0] > 0:
        equity = [round(start_val * c / closes[0], 2) for c in closes]
    else:
        equity = []

    # Extract per-trade returns from log
    parsed_trades = parse_trades_from_log(log_lines)
    trade_returns = []   # would need price data to compute exact returns; use log wins/losses
    win_lines  = [l for l in log_lines if "take-profit" in l.lower()]
    loss_lines = [l for l in log_lines if "stop-loss" in l.lower() or "trailing-stop" in l.lower()]
    trade_returns  = [abs(0.04)] * len(win_lines) + [-abs(0.02)] * len(loss_lines)

    metrics = compute_all(equity, trade_returns, log_lines)

    # Date labels for chart
    from datetime import datetime, timedelta
    today  = datetime.now()
    labels = [
        (today - timedelta(days=len(equity) - i - 1)).strftime("%b %d")
        for i in range(len(equity))
    ]

    return jsonify({
        "ticker"      : ticker,
        "labels"      : labels,
        "market_open" : is_market_hours(),
        **metrics,
    })


# ── /api/bots/pause ──────────────────────────────────────────────────────────
@app.route("/api/bots/pause", methods=["POST"])
def api_pause_bot():
    """Mark a bot as paused in state file (alpaca_bot.py checks this flag)."""
    data   = request.get_json()
    bot_id = data.get("id") if data else None
    bots   = load_bot_state()
    for b in bots:
        if b["id"] == bot_id:
            b["status"] = "paused"
            save_bot_state(bots)
            log.info(f"Bot {bot_id} paused via API")
            return jsonify({"ok": True, "status": "paused"})
    return jsonify({"error": "bot not found"}), 404


# ── /api/bots/resume ─────────────────────────────────────────────────────────
@app.route("/api/bots/resume", methods=["POST"])
def api_resume_bot():
    """Mark a bot as running."""
    data   = request.get_json()
    bot_id = data.get("id") if data else None
    bots   = load_bot_state()
    for b in bots:
        if b["id"] == bot_id:
            b["status"] = "running"
            save_bot_state(bots)
            log.info(f"Bot {bot_id} resumed via API")
            return jsonify({"ok": True, "status": "running"})
    return jsonify({"error": "bot not found"}), 404


# ── /api/bots/create ─────────────────────────────────────────────────────────
@app.route("/api/bots/create", methods=["POST"])
def api_create_bot():
    """
    Add a new bot to state and optionally launch alpaca_bot.py as a subprocess.
    Body: { name, ticker, gens, pop, alloc, stop }
    """
    data   = request.get_json()
    if not data:
        return jsonify({"error": "no body"}), 400

    ticker = data.get("ticker", "GLD")
    bots   = load_bot_state()
    new_id = f"b{int(time.time())}"
    new_bot = {
        "id"              : new_id,
        "name"            : data.get("name", f"{ticker} bot"),
        "ticker"          : ticker,
        "status"          : "running",
        "gens"            : int(data.get("gens", 200)),
        "pop"             : int(data.get("pop", 100)),
        "alloc"           : int(data.get("alloc", 40)),
        "stop"            : float(data.get("stop", 2.0)),
        "pid"             : None,
        "chromosome_file" : f"{ticker}_best_chromosome.csv",
        "log_file"        : f"{ticker}_bot.log",
    }
    bots.append(new_bot)
    save_bot_state(bots)
    log.info(f"Created bot {new_id} ({ticker})")
    return jsonify({"ok": True, "id": new_id, "bot": new_bot})


# ── /api/bots/delete ─────────────────────────────────────────────────────────
@app.route("/api/bots/delete", methods=["POST"])
def api_delete_bot():
    data   = request.get_json()
    bot_id = data.get("id") if data else None
    bots   = load_bot_state()
    before = len(bots)
    bots   = [b for b in bots if b["id"] != bot_id]
    if len(bots) == before:
        return jsonify({"error": "bot not found"}), 404
    save_bot_state(bots)
    return jsonify({"ok": True})


# ── /api/health ──────────────────────────────────────────────────────────────
@app.route("/api/health")
def api_health():
    """Quick liveness check — useful for the dashboard connection indicator."""
    alpaca_live = False
    if alpaca_ok():
        try:
            trading_client.get_clock()
            alpaca_live = True
        except Exception:
            pass
    return jsonify({
        "status"     : "ok",
        "alpaca"     : alpaca_live,
        "mode"       : "paper" if IS_PAPER else "live",
        "timestamp"  : datetime.now().isoformat(),
    })
@app.route("/api/tickers/search")
def api_ticker_search():
    """
    Search tickers for the bot-creation autocomplete.
    ?q=NVDA
    Returns popular matches instantly, or does a live yfinance lookup
    if the query looks like a valid ticker not in the popular list.
    """
    query = request.args.get("q", "")
    results = search_tickers(query, limit=8)
    return jsonify(results)
@app.route("/api/tickers/validate")
def api_ticker_validate():
    """
    Full validation of a specific ticker before bot creation.
    ?symbol=NVDA
    Returns volatility, asset type, and auto-tuned settings preview.
    """
    symbol = request.args.get("symbol", "")
    if not symbol:
        return jsonify({"error": "symbol required"}), 400

    ok, info = validate_ticker(symbol)
    if not ok:
        return jsonify({"valid": False, "error": info.get("error", "Unknown error")}), 200

    from ticker_manager import auto_tune_config
    tuned = auto_tune_config(info)

    return jsonify({
        "valid"          : True,
        "symbol"         : info["symbol"],
        "name"           : info["name"],
        "asset_type"     : info["asset_type"],
        "volatility"     : info["volatility"],
        "volatility_band": tuned["volatility_band"],
        "last_price"     : info["last_price"],
        "data_points"    : info["data_points"],
        "preview"        : {
            "max_allocation_pct": tuned["risk"]["max_allocation_pct"],
            "stop_loss_pct"     : tuned["risk"]["stop_loss_pct"],
            "take_profit_pct"   : tuned["risk"]["take_profit_pct"],
            "ga_generations"    : tuned["ga"]["generations"],
            "ga_population"     : tuned["ga"]["population_size"],
            "is_24_7"           : info["asset_type"] == "crypto",
        }
    })


# ── /api/tickers/popular ─────────────────────────────────────────────────────
@app.route("/api/tickers/popular")
def api_tickers_popular():
    """Returns the curated list shown before the user types anything."""
    return jsonify(POPULAR_TICKERS)


# ── /api/bots/create_dynamic ─────────────────────────────────────────────────
@app.route("/api/bots/create_dynamic", methods=["POST"])
def api_create_bot_dynamic():
    """
    Create a bot for ANY validated ticker.
    Body: { "symbol": "NVDA", "name": "My NVDA bot" (optional) }

    This validates the ticker, classifies it, auto-tunes GA settings
    based on volatility, and saves the config. The bot starts in
    'pending_training' status — call /api/bots/train to kick off the
    GA training, which the bot_manager will then pick up.
    """
    data = request.get_json()
    if not data or "symbol" not in data:
        return jsonify({"error": "symbol required"}), 400

    symbol = data["symbol"].strip().upper()
    name   = data.get("name")

    ok, result = create_bot_config(symbol, name)
    if not ok:
        return jsonify({"error": result.get("error", "Validation failed")}), 400

    add_bot_config(result)
    log.info(f"Created dynamic bot config: {result['id']} ({symbol})")

    return jsonify({
        "ok"     : True,
        "id"     : result["id"],
        "bot"    : result,
        "message": f"Bot created for {symbol}. Training required before it can trade."
    })


# ── /api/bots/train ──────────────────────────────────────────────────────────
@app.route("/api/bots/train", methods=["POST"])
def api_train_bot():
    """
    Kicks off GA training for a pending bot as a background subprocess.
    Body: { "id": "bot_1234567890" }
    Training runs async — poll /api/bots/train_status to check progress.
    """
    data   = request.get_json()
    bot_id = data.get("id") if data else None

    configs = load_all_configs()
    if bot_id not in configs:
        return jsonify({"error": "bot not found"}), 404

    config = configs[bot_id]
    ticker = config["ticker"]

    update_bot_status(bot_id, "training")

    def _run_training():
        try:
            import subprocess, json as json_lib

            tmp_config = Path(f"_train_config_{bot_id}.json")
            tmp_config.write_text(json_lib.dumps(config))

            import sys
            result = subprocess.run(
                [sys.executable, "train_dynamic.py", "--config", str(tmp_config)],
                capture_output=True, text=True, timeout=1800,
                cwd=str(Path(__file__).parent)
            )
            tmp_config.unlink(missing_ok=True)

            if result.returncode == 0:
                update_bot_status(bot_id, "running")
                log.info(f"Training complete for {bot_id} ({ticker})")
            else:
                update_bot_status(bot_id, "training_failed")
                log.error(f"Training failed for {bot_id}: {result.stderr[-500:]}")

        except Exception as e:
            update_bot_status(bot_id, "training_failed")
            log.error(f"Training subprocess error for {bot_id}: {e}")

    thread = threading.Thread(target=_run_training, daemon=True)
    thread.start()

    return jsonify({"ok": True, "status": "training", "message": "Training started in background"})


# ── /api/bots/train_status ───────────────────────────────────────────────────
@app.route("/api/bots/train_status")
def api_train_status():
    """Check training status for a bot. ?id=bot_1234567890"""
    bot_id  = request.args.get("id", "")
    configs = load_all_configs()
    bot     = configs.get(bot_id)
    if not bot:
        return jsonify({"error": "bot not found"}), 404
    return jsonify({"id": bot_id, "status": bot["status"]})


# ── /api/bots/all ─────────────────────────────────────────────────────────────
@app.route("/api/bots/all")
def api_bots_all():
    """All dynamically configured bots, including pending/training ones."""
    configs = load_all_configs()
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
    return jsonify(result)


# ── /api/bots/delete_dynamic ─────────────────────────────────────────────────
@app.route("/api/bots/delete_dynamic", methods=["POST"])
def api_delete_bot_dynamic():
    """Remove a dynamically created bot config. Body: { "id": "bot_..." }"""
    data   = request.get_json()
    bot_id = data.get("id") if data else None
    if not bot_id:
        return jsonify({"error": "id required"}), 400
    ok = remove_bot_config(bot_id)
    if not ok:
        return jsonify({"error": "bot not found"}), 404
    return jsonify({"ok": True})
# ── /api/auth/register ───────────────────────────────────────────────────────
@app.route("/api/auth/register", methods=["POST"])
def api_register():
    """Register a new user. Body: { username, password, email? }"""
    data     = request.get_json()
    username = (data or {}).get("username", "").strip()
    password = (data or {}).get("password", "")
    email    = (data or {}).get("email", "").strip() or None
 
    ok, result = register_user(username, password, email)
    if not ok:
        return jsonify({"error": result}), 400
 
    # result is the session token
    return jsonify({
        "ok"      : True,
        "token"   : result,
        "username": username,
        "message" : "Account created! Welcome to BotFire.",
        "balance" : 100000.0,
    })
 
 
# ── /api/auth/login ──────────────────────────────────────────────────────────
@app.route("/api/auth/login", methods=["POST"])
def api_login():
    """Login. Body: { username, password }"""
    data     = request.get_json()
    username = (data or {}).get("username", "").strip()
    password = (data or {}).get("password", "")
 
    ok, result = login_user(username, password)
    if not ok:
        return jsonify({"error": result}), 401
 
    return jsonify({
        "ok"      : True,
        "token"   : result,
        "username": username,
        "message" : "Welcome back!",
    })
 
 
# ── /api/auth/logout ─────────────────────────────────────────────────────────
@app.route("/api/auth/logout", methods=["POST"])
def api_logout():
    """Logout — invalidate session token."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if token:
        delete_session(token)
    return jsonify({"ok": True})
 
 
# ── /api/auth/me ─────────────────────────────────────────────────────────────
@app.route("/api/auth/me")
@require_auth
def api_me():
    """Get current user's profile."""
    profile = get_user_profile(g.current_user["id"])
    return jsonify(profile)
 
 
# ── /api/global/stats ────────────────────────────────────────────────────────
@app.route("/api/global/stats")
def api_global_stats():
    """Public — global platform stats for the fire counter."""
    stats = get_global_stats()
    return jsonify(stats)
 
 
# ── /api/admin/users ─────────────────────────────────────────────────────────
@app.route("/api/admin/users")
@require_admin
def api_admin_users():
    """Admin only — list all users."""
    page = int(request.args.get("page", 1))
    users = get_all_users(page=page)
    return jsonify(users)
 
 
# ── /api/user/bots ───────────────────────────────────────────────────────────
@app.route("/api/user/bots")
@require_auth
def api_user_bots():
    """Get current user's bots."""
    bots = get_user_bots(g.current_user["id"])
    return jsonify(bots)
 
 
# ── /api/user/bots/add ───────────────────────────────────────────────────────
@app.route("/api/user/bots/add", methods=["POST"])
@require_auth
def api_user_add_bot():
    """Add a bot to the current user's account."""
    data          = request.get_json()
    bot_config_id = data.get("bot_config_id")
    ticker        = data.get("ticker")
    name          = data.get("name", f"{ticker} bot")
 
    if not bot_config_id or not ticker:
        return jsonify({"error": "bot_config_id and ticker required"}), 400
 
    bot_id = add_user_bot(g.current_user["id"], bot_config_id, ticker, name)
    return jsonify({"ok": True, "id": bot_id})

register_user_routes(app)

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Step 5 — Flask API server")
    print("="*55)
    print(f"  Mode    : {'PAPER' if IS_PAPER else 'LIVE'}")
    print(f"  API key : {API_KEY[:6]}{'*'*10 if API_KEY else ' (MISSING)'}")
    print(f"  Running : http://localhost:5000")
    print(f"\n  Endpoints:")
    print(f"    GET  /api/health")
    print(f"    GET  /api/account")
    print(f"    GET  /api/positions")
    print(f"    GET  /api/bots")
    print(f"    GET  /api/logs?bot_id=b1")
    print(f"    GET  /api/performance?ticker=GLD&days=90")
    print(f"    POST /api/bots/pause    {{\"id\":\"b1\"}}")
    print(f"    POST /api/bots/resume   {{\"id\":\"b1\"}}")
    print(f"    POST /api/bots/create   {{...config}}")
    print(f"    POST /api/bots/delete   {{\"id\":\"b1\"}}")
    print("="*55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)

