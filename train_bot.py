"""
train_bot.py  —  Train GA chromosome for any ticker
=====================================================
Run before bot_manager.py to generate chromosomes.

Usage
-----
python train_bot.py --ticker GLD          # train one
python train_bot.py --ticker SPY
python train_bot.py --ticker BTC-USD
python train_bot.py --all                 # train all three
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from stock_data import download_stock_data, add_indicators, preprocess_data
from genetic_algorithm import run_ga, save_results
from bot_manager import BOT_CONFIGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s"
)
log = logging.getLogger(__name__)


def train(ticker: str) -> None:
    cfg = BOT_CONFIGS[ticker]
    log.info(f"\n{'='*55}")
    log.info(f"  Training: {ticker}")
    log.info(f"  Generations : {cfg['ga']['generations']}")
    log.info(f"  Population  : {cfg['ga']['population_size']}")
    log.info(f"{'='*55}")

    end = datetime.now().strftime("%Y-%m-%d")

    log.info(f"Downloading {ticker} data from {cfg['start_date']} to {end}...")
    df_raw = download_stock_data(ticker, cfg["start_date"], end)
    df_raw.to_csv(f"{ticker}_raw.csv")

    log.info("Computing indicators...")
    df_ind = add_indicators(df_raw.copy())

    log.info("Preprocessing...")
    df_scaled, scaler = preprocess_data(df_ind)
    df_scaled.to_csv(f"{ticker}_scaled.csv")

    # Align
    idx       = df_scaled.index.intersection(df_raw.index)
    df_scaled = df_scaled.loc[idx]
    df_raw_al = df_raw.loc[idx]

    log.info(f"Running GA on {len(df_scaled)} rows × {df_scaled.shape[1]} features...")
    result = run_ga(df_scaled, df_raw_al, cfg["ga"])
    save_results(result, ticker)

    log.info(f"\n[{ticker}] Training complete")
    log.info(f"  Best fitness  : {result['best_fitness']:.4f}")
    log.info(f"  Total return  : {result['best_stats']['total_return']:+.2%}")
    log.info(f"  Win rate      : {result['best_stats']['win_rate']:.2%}")
    log.info(f"  Trades        : {result['best_stats']['n_trades']}")
    log.info(f"  Max drawdown  : {result['best_stats']['max_drawdown']:.2%}")
    log.info(f"  Saved to      : {cfg['chromosome_file']}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", help="Ticker to train (GLD, SPY, BTC-USD)")
    parser.add_argument("--all", action="store_true", help="Train all three bots")
    args = parser.parse_args()

    if args.all:
        tickers = list(BOT_CONFIGS.keys())
    elif args.ticker:
        if args.ticker not in BOT_CONFIGS:
            print(f"Unknown ticker. Options: {list(BOT_CONFIGS.keys())}")
            return
        tickers = [args.ticker]
    else:
        parser.print_help()
        return

    for ticker in tickers:
        path = Path(BOT_CONFIGS[ticker]["chromosome_file"])
        if path.exists():
            overwrite = input(f"{path} already exists. Retrain? [y/N]: ").strip().lower()
            if overwrite != "y":
                log.info(f"Skipping {ticker}")
                continue
        train(ticker)

    print("\nAll done. Run: python bot_manager.py")


if __name__ == "__main__":
    main()
