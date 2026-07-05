"""
train_dynamic.py  --  Train a single bot from a JSON config
=============================================================
Called by api.py as a subprocess when a user creates a bot for a
new ticker via the dashboard UI. Unlike train_bot.py (which uses
the hardcoded BOT_CONFIGS dict), this reads a config dict directly
from a JSON file -- supporting ANY validated ticker.

Usage
-----
python train_dynamic.py --config _train_config_bot_1234567890.json
"""

import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from stock_data import download_stock_data, add_indicators, preprocess_data
from genetic_algorithm import run_ga, save_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s"
)
log = logging.getLogger(__name__)


def train_from_config(config: dict) -> bool:
    ticker = config["ticker"]
    log.info(f"\n{'='*55}")
    log.info(f"  Dynamic training: {ticker} ({config.get('display_name', ticker)})")
    log.info(f"  Asset type: {config.get('asset_type', 'unknown')}")
    log.info(f"  Volatility band: {config.get('volatility_band', 'unknown')}")
    log.info(f"  Generations : {config['ga']['generations']}")
    log.info(f"  Population  : {config['ga']['population_size']}")
    log.info(f"{'='*55}")

    yf_ticker  = ticker.replace("/", "-")
    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = config.get("start_date", "2020-01-01")

    try:
        log.info(f"Downloading {yf_ticker} data from {start_date} to {end_date}...")
        df_raw = download_stock_data(yf_ticker, start_date, end_date)
        df_raw.to_csv(f"{yf_ticker}_raw.csv")

        log.info("Computing indicators...")
        df_ind = add_indicators(df_raw.copy())

        log.info("Preprocessing...")
        df_scaled, scaler = preprocess_data(df_ind)
        df_scaled.to_csv(f"{yf_ticker}_scaled.csv")

        idx       = df_scaled.index.intersection(df_raw.index)
        df_scaled = df_scaled.loc[idx]
        df_raw_al = df_raw.loc[idx]

        if len(df_scaled) < 100:
            log.error(f"Insufficient data after preprocessing: {len(df_scaled)} rows")
            return False

        log.info(f"Running GA on {len(df_scaled)} rows x {df_scaled.shape[1]} features...")
        result = run_ga(df_scaled, df_raw_al, config["ga"])
        save_results(result, yf_ticker)

        log.info(f"\n[{ticker}] Training complete")
        log.info(f"  Best fitness  : {result['best_fitness']:.4f}")
        log.info(f"  Total return  : {result['best_stats']['total_return']:+.2%}")
        log.info(f"  Win rate      : {result['best_stats']['win_rate']:.2%}")
        log.info(f"  Trades        : {result['best_stats']['n_trades']}")
        log.info(f"  Max drawdown  : {result['best_stats']['max_drawdown']:.2%}")
        log.info(f"  Saved to      : {yf_ticker}_best_chromosome.csv\n")

        return True

    except Exception as e:
        log.error(f"Training failed: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to bot config JSON file")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        log.error(f"Config file not found: {config_path}")
        sys.exit(1)

    config  = json.loads(config_path.read_text())
    success = train_from_config(config)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
