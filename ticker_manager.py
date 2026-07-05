"""
ticker_manager.py  —  Dynamic Ticker Support
==============================================
Replaces the hardcoded BOT_CONFIGS dict in bot_manager.py with a
dynamic system that:
  1. Validates any ticker symbol against yfinance
  2. Classifies it as stock, ETF, or crypto
  3. Auto-tunes GA hyperparameters and risk limits based on the
     asset's historical volatility
  4. Persists bot configs to bot_configs.json (replaces the static dict)

Usage
-----
from ticker_manager import validate_ticker, create_bot_config, load_all_configs

ok, info = validate_ticker("NVDA")
ok, config = create_bot_config("NVDA")
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

CONFIG_FILE = Path(__file__).parent / "bot_configs.json"

CRYPTO_SUFFIXES = {"-USD", "/USD", "-USDT", "/USDT"}
KNOWN_CRYPTO    = {"BTC", "ETH", "LTC", "BCH", "SOL", "DOGE", "ADA", "XRP"}

# Popular tickers shown in the UI search before the user types anything
POPULAR_TICKERS = [
    {"symbol": "GLD",      "name": "SPDR Gold Shares",         "type": "etf"},
    {"symbol": "SLV",      "name": "iShares Silver Trust",     "type": "etf"},
    {"symbol": "SPY",      "name": "SPDR S&P 500 ETF",         "type": "etf"},
    {"symbol": "QQQ",      "name": "Invesco QQQ (Nasdaq-100)", "type": "etf"},
    {"symbol": "DIA",      "name": "SPDR Dow Jones ETF",       "type": "etf"},
    {"symbol": "USO",      "name": "United States Oil Fund",   "type": "etf"},
    {"symbol": "XLE",      "name": "Energy Select Sector ETF", "type": "etf"},
    {"symbol": "AAPL",     "name": "Apple Inc.",               "type": "stock"},
    {"symbol": "MSFT",     "name": "Microsoft Corp.",          "type": "stock"},
    {"symbol": "NVDA",     "name": "NVIDIA Corp.",             "type": "stock"},
    {"symbol": "TSLA",     "name": "Tesla Inc.",               "type": "stock"},
    {"symbol": "AMZN",     "name": "Amazon.com Inc.",          "type": "stock"},
    {"symbol": "GOOGL",    "name": "Alphabet Inc.",            "type": "stock"},
    {"symbol": "META",     "name": "Meta Platforms Inc.",      "type": "stock"},
    {"symbol": "BTC/USD",  "name": "Bitcoin",                  "type": "crypto"},
    {"symbol": "ETH/USD",  "name": "Ethereum",                 "type": "crypto"},
    {"symbol": "SOL/USD",  "name": "Solana",                   "type": "crypto"},
    {"symbol": "DOGE/USD", "name": "Dogecoin",                 "type": "crypto"},
]


# ──────────────────────────────────────────────────────────────────────────────
# TICKER CLASSIFICATION
# ──────────────────────────────────────────────────────────────────────────────

def classify_ticker(symbol: str) -> str:
    """Returns 'crypto', 'etf', or 'stock' based on the symbol."""
    base = symbol.split("/")[0].split("-")[0].upper()
    if "/" in symbol or any(symbol.upper().endswith(s) for s in CRYPTO_SUFFIXES):
        return "crypto"
    if base in KNOWN_CRYPTO:
        return "crypto"
    known_etfs = {"GLD","SLV","SPY","QQQ","DIA","USO","XLE","XLF","XLK",
                  "IAU","VOO","VTI","ARKK","SOXL","TQQQ","SQQQ","UVXY"}
    if base in known_etfs:
        return "etf"
    return "stock"


# ──────────────────────────────────────────────────────────────────────────────
# TICKER VALIDATION  — confirms the symbol has tradeable data
# ──────────────────────────────────────────────────────────────────────────────

def validate_ticker(symbol: str) -> tuple[bool, dict]:
    """
    Validates a ticker exists and has sufficient historical data.

    Returns (is_valid, info_dict) where info_dict contains:
      symbol, name, asset_type, volatility, avg_volume,
      data_points, error (if invalid)
    """
    symbol = symbol.strip().upper()
    asset_type = classify_ticker(symbol)

    try:
        import yfinance as yf
        yf_symbol = symbol.replace("/", "-")

        ticker = yf.Ticker(yf_symbol)
        hist   = ticker.history(period="6mo")

        if hist.empty or len(hist) < 30:
            return False, {
                "symbol": symbol,
                "error": f"Insufficient data ({len(hist)} days) — "
                         f"need at least 30 trading days"
            }

        returns    = hist["Close"].pct_change().dropna()
        volatility = float(returns.std() * np.sqrt(252))
        avg_volume = float(hist["Volume"].mean()) if "Volume" in hist else 0

        try:
            info = ticker.info
            name = info.get("longName") or info.get("shortName") or symbol
        except Exception:
            name = symbol

        return True, {
            "symbol"      : symbol,
            "name"        : name,
            "asset_type"  : asset_type,
            "volatility"  : round(volatility, 4),
            "avg_volume"  : int(avg_volume),
            "data_points" : len(hist),
            "last_price"  : round(float(hist["Close"].iloc[-1]), 2),
        }

    except Exception as e:
        return False, {"symbol": symbol, "error": str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# AUTO-TUNED GA CONFIG  — based on asset type + measured volatility
# ──────────────────────────────────────────────────────────────────────────────

def auto_tune_config(info: dict) -> dict:
    """
    Generates GA hyperparameters and risk limits automatically based on
    asset type and measured historical volatility.

    Volatility bands (annualised):
      Low    < 0.20   (e.g. GLD, large-cap blue chips)
      Medium 0.20-0.40 (e.g. SPY, most large caps)
      High   0.40-0.70 (e.g. growth stocks, small caps)
      Extreme > 0.70   (e.g. crypto, meme stocks)
    """
    vol        = info.get("volatility", 0.25)
    asset_type = info.get("asset_type", "stock")

    if vol < 0.20:
        band = "low"
    elif vol < 0.40:
        band = "medium"
    elif vol < 0.70:
        band = "high"
    else:
        band = "extreme"

    ga_presets = {
        "low": {
            "population_size": 100, "generations": 200, "elite_count": 5,
            "tournament_size": 5, "crossover_rate": 0.85,
            "mutation_rate_init": 0.12, "mutation_rate_min": 0.01,
            "mutation_rate_max": 0.25, "mutation_step": 0.12,
            "stagnation_window": 20, "fitness_alpha": 0.60, "fitness_beta": 0.40,
            "drawdown_penalty": 0.50, "max_drawdown_thresh": 0.15,
            "min_trades": 12, "random_seed": 42,
        },
        "medium": {
            "population_size": 120, "generations": 220, "elite_count": 6,
            "tournament_size": 5, "crossover_rate": 0.85,
            "mutation_rate_init": 0.15, "mutation_rate_min": 0.01,
            "mutation_rate_max": 0.30, "mutation_step": 0.15,
            "stagnation_window": 20, "fitness_alpha": 0.55, "fitness_beta": 0.45,
            "drawdown_penalty": 0.55, "max_drawdown_thresh": 0.20,
            "min_trades": 10, "random_seed": 42,
        },
        "high": {
            "population_size": 140, "generations": 260, "elite_count": 7,
            "tournament_size": 6, "crossover_rate": 0.82,
            "mutation_rate_init": 0.18, "mutation_rate_min": 0.02,
            "mutation_rate_max": 0.35, "mutation_step": 0.17,
            "stagnation_window": 18, "fitness_alpha": 0.50, "fitness_beta": 0.50,
            "drawdown_penalty": 0.45, "max_drawdown_thresh": 0.22,
            "min_trades": 9, "random_seed": 42,
        },
        "extreme": {
            "population_size": 150, "generations": 300, "elite_count": 8,
            "tournament_size": 6, "crossover_rate": 0.80,
            "mutation_rate_init": 0.20, "mutation_rate_min": 0.02,
            "mutation_rate_max": 0.40, "mutation_step": 0.18,
            "stagnation_window": 15, "fitness_alpha": 0.50, "fitness_beta": 0.50,
            "drawdown_penalty": 0.40, "max_drawdown_thresh": 0.25,
            "min_trades": 8, "random_seed": 42,
        },
    }

    risk_presets = {
        "low": {
            "min_allocation_pct": 0.05, "max_allocation_pct": 0.40,
            "weight_threshold": 0.28, "stop_loss_pct": 0.015,
            "take_profit_pct": 0.03, "trailing_stop_pct": 0.012,
        },
        "medium": {
            "min_allocation_pct": 0.05, "max_allocation_pct": 0.35,
            "weight_threshold": 0.30, "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04, "trailing_stop_pct": 0.015,
        },
        "high": {
            "min_allocation_pct": 0.04, "max_allocation_pct": 0.28,
            "weight_threshold": 0.33, "stop_loss_pct": 0.025,
            "take_profit_pct": 0.05, "trailing_stop_pct": 0.02,
        },
        "extreme": {
            "min_allocation_pct": 0.03, "max_allocation_pct": 0.20,
            "weight_threshold": 0.38, "stop_loss_pct": 0.035,
            "take_profit_pct": 0.07, "trailing_stop_pct": 0.028,
        },
    }

    is_crypto = asset_type == "crypto"

    return {
        "volatility_band"     : band,
        "ga"                  : ga_presets[band],
        "risk"                : risk_presets[band],
        "market_open_delay_s" : 0 if is_crypto else 300,
        "intraday_interval_s" : 120 if is_crypto else (45 if band in ("high","extreme") else 60),
        "lookback_days"       : 90 if is_crypto else 60,
        "start_date"          : "2021-01-01" if is_crypto else "2020-01-01",
    }


# ──────────────────────────────────────────────────────────────────────────────
# BOT CONFIG CREATION  — combines validation + auto-tuning
# ──────────────────────────────────────────────────────────────────────────────

def create_bot_config(symbol: str, name: str = None) -> tuple[bool, dict]:
    """
    Full pipeline: validate ticker → classify → auto-tune → build config.
    Returns (success, config_or_error).
    """
    ok, info = validate_ticker(symbol)
    if not ok:
        return False, info

    tuned = auto_tune_config(info)
    symbol_clean = info["symbol"]
    chromosome_ticker = symbol_clean.replace("/", "-")

    config = {
        "id"                 : f"bot_{int(datetime.now().timestamp())}",
        "name"               : name or f"{symbol_clean} bot",
        "ticker"             : symbol_clean,
        "display_name"       : info["name"],
        "asset_type"         : info["asset_type"],
        "volatility"         : info["volatility"],
        "volatility_band"    : tuned["volatility_band"],
        "chromosome_file"    : f"{chromosome_ticker}_best_chromosome.csv",
        "log_file"           : f"{chromosome_ticker}_bot.log",
        "status"             : "pending_training",
        "created_at"         : datetime.now().isoformat(),
        "ga"                 : tuned["ga"],
        "risk"               : tuned["risk"],
        "market_open_delay_s": tuned["market_open_delay_s"],
        "intraday_interval_s": tuned["intraday_interval_s"],
        "lookback_days"      : tuned["lookback_days"],
        "start_date"         : tuned["start_date"],
    }

    return True, config


# ──────────────────────────────────────────────────────────────────────────────
# PERSISTENCE  — bot_configs.json replaces the hardcoded BOT_CONFIGS dict
# ──────────────────────────────────────────────────────────────────────────────

def load_all_configs() -> dict:
    """Load all bot configs from bot_configs.json. Seeds defaults if missing."""
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())

    defaults = {}
    for symbol, name in [("GLD", "GLD bot"), ("SPY", "SPY bot"), ("BTC/USD", "BTC bot")]:
        ok, cfg = create_bot_config(symbol, name)
        if ok:
            cfg["status"] = "running"
            defaults[cfg["id"]] = cfg
    save_all_configs(defaults)
    return defaults


def save_all_configs(configs: dict) -> None:
    CONFIG_FILE.write_text(json.dumps(configs, indent=2))


def add_bot_config(config: dict) -> None:
    configs = load_all_configs()
    configs[config["id"]] = config
    save_all_configs(configs)
    log.info(f"Added bot config: {config['id']} ({config['ticker']})")


def remove_bot_config(bot_id: str) -> bool:
    configs = load_all_configs()
    if bot_id in configs:
        del configs[bot_id]
        save_all_configs(configs)
        return True
    return False


def get_bot_config(bot_id: str) -> dict | None:
    return load_all_configs().get(bot_id)


def update_bot_status(bot_id: str, status: str) -> None:
    configs = load_all_configs()
    if bot_id in configs:
        configs[bot_id]["status"] = status
        save_all_configs(configs)


# ──────────────────────────────────────────────────────────────────────────────
# SEARCH  — used by the dashboard's ticker autocomplete
# ──────────────────────────────────────────────────────────────────────────────

def search_tickers(query: str, limit: int = 8) -> list[dict]:
    """
    Search popular tickers first (instant), then fall back to a live
    yfinance lookup if the query looks like a valid symbol not in the
    popular list.
    """
    query = query.strip().upper()
    if not query:
        return POPULAR_TICKERS[:limit]

    matches = [
        t for t in POPULAR_TICKERS
        if query in t["symbol"].upper() or query in t["name"].upper()
    ]

    if matches:
        return matches[:limit]

    if 1 <= len(query) <= 6 and query.replace("/", "").replace("-", "").isalpha():
        ok, info = validate_ticker(query)
        if ok:
            return [{
                "symbol": info["symbol"],
                "name"  : info["name"],
                "type"  : info["asset_type"],
                "live_lookup": True,
            }]

    return []


# ──────────────────────────────────────────────────────────────────────────────
# CLI — test ticker validation
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
        print(f"\nValidating {symbol}...")
        ok, info = validate_ticker(symbol)
        if ok:
            print(f"  Valid: {info['name']} ({info['asset_type']})")
            print(f"  Volatility: {info['volatility']:.2%} annualised")
            print(f"  Last price: ${info['last_price']}")
            print(f"  Data points: {info['data_points']}")

            tuned = auto_tune_config(info)
            print(f"\n  Auto-tuned band: {tuned['volatility_band']}")
            print(f"  GA: {tuned['ga']['generations']} gen x "
                  f"{tuned['ga']['population_size']} pop")
            print(f"  Max allocation: {tuned['risk']['max_allocation_pct']*100:.0f}%")
            print(f"  Stop-loss: {tuned['risk']['stop_loss_pct']*100:.1f}%")
        else:
            print(f"  Invalid: {info.get('error')}")
    else:
        print("Usage: python ticker_manager.py SYMBOL")
        print("Example: python ticker_manager.py NVDA")
