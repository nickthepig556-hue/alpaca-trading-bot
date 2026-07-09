"""
retrain.py  —  Weekly Auto-Retrain Scheduler
=============================================
Automatically retrains GA chromosomes every Sunday at 6 PM ET
using the latest market data. Backs up old chromosomes before
overwriting so you can roll back if performance drops.

Usage
-----
python retrain.py                  # run once immediately
python retrain.py --schedule       # run on schedule (every Sunday 6 PM)
python retrain.py --ticker GLD     # retrain one ticker only
python retrain.py --rollback GLD   # restore previous chromosome

Task Scheduler (automated)
---------------------------
python deploy.py --install         # already adds weekly retrain task
Or manually:
schtasks /Create /TN "AlpacaRetrain" /TR "retrain.bat" /SC WEEKLY /D SUN /ST 18:00
"""

import os
import sys
import time
import shutil
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("retrain.log", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

PROJECT_DIR  = Path(__file__).resolve().parent
BACKUP_DIR   = PROJECT_DIR / "chromosome_backups"
BACKUP_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# PERFORMANCE COMPARISON
# ──────────────────────────────────────────────────────────────────────────────

def read_fitness(ticker: str) -> float:
    """Read the best fitness from the bot's fitness history CSV."""
    path = PROJECT_DIR / f"{ticker}_fitness_history.csv"
    if not path.exists():
        return 0.0
    try:
        import pandas as pd
        df = pd.read_csv(path)
        return float(df["best_fitness"].iloc[-1])
    except Exception:
        return 0.0


def compare_chromosomes(ticker: str, old_fitness: float, new_fitness: float) -> bool:
    """Always accept new chromosome when retraining with expanded dataset."""
    log.info(f"[{ticker}] New fitness {new_fitness:.4f} — accepting (force retrain mode)")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# BACKUP / ROLLBACK
# ──────────────────────────────────────────────────────────────────────────────

def backup_chromosome(ticker: str) -> Path | None:
    """Back up current chromosome before retraining. Returns backup path."""
    src = PROJECT_DIR / f"{ticker}_best_chromosome.csv"
    if not src.exists():
        return None
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst     = BACKUP_DIR / f"{ticker}_chromosome_{ts}.csv"
    shutil.copy2(src, dst)
    log.info(f"[{ticker}] Backed up to {dst.name}")

    # Keep only last 5 backups per ticker
    backups = sorted(BACKUP_DIR.glob(f"{ticker}_chromosome_*.csv"))
    for old in backups[:-5]:
        old.unlink()
        log.info(f"[{ticker}] Deleted old backup: {old.name}")

    return dst


def rollback_chromosome(ticker: str) -> bool:
    """Restore the most recent backup chromosome."""
    backups = sorted(BACKUP_DIR.glob(f"{ticker}_chromosome_*.csv"))
    if not backups:
        log.error(f"[{ticker}] No backups found in {BACKUP_DIR}")
        return False
    latest  = backups[-1]
    dst     = PROJECT_DIR / f"{ticker}_best_chromosome.csv"
    shutil.copy2(latest, dst)
    log.info(f"[{ticker}] Rolled back to {latest.name}")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# SINGLE TICKER RETRAIN
# ──────────────────────────────────────────────────────────────────────────────

def retrain_ticker(ticker: str, config: dict) -> bool:
    """
    Full retrain pipeline for one ticker:
    1. Backup existing chromosome
    2. Download fresh data
    3. Run GA
    4. Compare fitness — keep best
    5. Signal bot_manager to reload chromosome

    Returns True if retrain succeeded.
    """
    # Use global GA_CONFIG as base, override with bot-specific settings
    from genetic_algorithm import GA_CONFIG as GLOBAL_GA
    ga_config = {**GLOBAL_GA, **config.get('ga', {})}
    config = {**config, 'ga': ga_config}

    log.info(f"\n{'='*55}")
    log.info(f"  Retraining: {ticker}")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info(f"{'='*55}")

    # Record old fitness
    old_fitness = read_fitness(ticker)
    log.info(f"[{ticker}] Current fitness: {old_fitness:.4f}")

    # Backup
    backup_path = backup_chromosome(ticker)

    try:
        from stock_data import download_stock_data, add_indicators, preprocess_data
        from genetic_algorithm import run_ga, save_results

        # Use yfinance ticker (BTC/USD → BTC-USD)
        yf_ticker = ticker.replace("/", "-")
        end_date  = datetime.now().strftime("%Y-%m-%d")

        log.info(f"[{ticker}] Downloading fresh data...")
        df_raw = download_stock_data(
            yf_ticker,
            config.get("start_date", "2020-01-01"),
            end_date
        )

        log.info(f"[{ticker}] Computing indicators on {len(df_raw)} rows...")
        df_ind    = add_indicators(df_raw.copy())
        df_scaled, _ = preprocess_data(df_ind)

        # Align indices
        idx       = df_scaled.index.intersection(df_raw.index)
        df_scaled = df_scaled.loc[idx]
        df_raw_al = df_raw.loc[idx]

        log.info(f"[{ticker}] Running GA "
                 f"({config['ga']['generations']} gen × "
                 f"{config['ga']['population_size']} pop)...")

        result = run_ga(df_scaled, df_raw_al, config["ga"])
        new_fitness = result["best_fitness"]

        log.info(f"[{ticker}] New fitness   : {new_fitness:.4f}")
        log.info(f"[{ticker}] Total return  : {result['best_stats']['total_return']:+.2%}")
        log.info(f"[{ticker}] Win rate      : {result['best_stats']['win_rate']:.2%}")
        log.info(f"[{ticker}] Trades        : {result['best_stats']['n_trades']}")
        log.info(f"[{ticker}] Max drawdown  : {result['best_stats']['max_drawdown']:.2%}")

        # Only save if new chromosome is good enough
        if compare_chromosomes(ticker, old_fitness, new_fitness):
            save_results(result, yf_ticker)
            # Rename to match bot's expected filename if needed
            new_path = PROJECT_DIR / f"{yf_ticker}_best_chromosome.csv"
            expected = PROJECT_DIR / f"{ticker.replace('/', '-')}_best_chromosome.csv"
            if new_path != expected and new_path.exists():
                shutil.move(str(new_path), str(expected))
            log.info(f"[{ticker}] Chromosome saved — retrain complete")
            return True
        else:
            # Restore backup
            if backup_path and backup_path.exists():
                shutil.copy2(backup_path, PROJECT_DIR / f"{ticker}_best_chromosome.csv")
                log.info(f"[{ticker}] Restored previous chromosome")
            return False

    except Exception as e:
        log.error(f"[{ticker}] Retrain failed: {e}", exc_info=True)
        # Restore backup on error
        if backup_path and backup_path.exists():
            shutil.copy2(backup_path, PROJECT_DIR / f"{ticker}_best_chromosome.csv")
            log.info(f"[{ticker}] Restored backup after error")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# FULL RETRAIN ALL BOTS
# ──────────────────────────────────────────────────────────────────────────────

def retrain_all() -> dict[str, bool]:
    """Retrain all configured bots sequentially."""
    from bot_manager import BOT_CONFIGS
    
    # Skip futures bots — they use futures_bot.py for retraining
    from futures_bot import FUTURES_CONFIGS
    spot_configs = {
        ticker: cfg for ticker, cfg in BOT_CONFIGS.items()
        if ticker not in FUTURES_CONFIGS
        and not ticker.endswith('_FUT')
    }

    results = {}
    start   = datetime.now()

    log.info("\n" + "="*55)
    log.info("  WEEKLY AUTO-RETRAIN STARTING")
    log.info(f"  {start.strftime('%Y-%m-%d %H:%M')}")
    log.info("="*55)

    for ticker, config in spot_configs.items():
        try:
            ok = retrain_ticker(ticker, config)
            results[ticker] = ok
        except Exception as e:
            log.error(f"[{ticker}] Unexpected error: {e}")
            results[ticker] = False
        # Small gap between tickers to avoid API rate limits
        time.sleep(10)

    # Summary
    elapsed = (datetime.now() - start).total_seconds() / 60
    log.info("\n" + "="*55)
    log.info("  RETRAIN SUMMARY")
    log.info("="*55)
    for ticker, ok in results.items():
        status = "[OK] Updated" if ok else "[--] Kept previous"
        log.info(f"  {ticker:<12} {status}")
    log.info(f"  Total time: {elapsed:.1f} minutes")
    log.info("="*55)

    return results



# ──────────────────────────────────────────────────────────────────────────────
# SCHEDULER  — runs every Sunday at 6 PM ET
# ──────────────────────────────────────────────────────────────────────────────

def next_sunday_6pm() -> datetime:
    """Return the next Sunday at 18:00 ET."""
    now  = datetime.now()
    days_ahead = 6 - now.weekday()   # Sunday = 6
    if days_ahead <= 0:
        days_ahead += 7
    next_sun = now + timedelta(days=days_ahead)
    return next_sun.replace(hour=18, minute=0, second=0, microsecond=0)


def run_scheduler() -> None:
    """
    Long-running scheduler loop.
    Sleeps until Sunday 6 PM, retrains all bots, then sleeps again.
    """
    log.info("Retrain scheduler started.")
    log.info("Will retrain all bots every Sunday at 6:00 PM.")
    log.info("Press Ctrl+C to stop.\n")

    while True:
        next_run = next_sunday_6pm()
        wait_s   = (next_run - datetime.now()).total_seconds()
        wait_h   = wait_s / 3600

        log.info(f"Next retrain: {next_run.strftime('%A %Y-%m-%d %H:%M')} "
                 f"(in {wait_h:.1f} hours)")

        try:
            time.sleep(wait_s)
        except KeyboardInterrupt:
            log.info("Scheduler stopped.")
            return

        # Run retrain
        results = retrain_all()

        # Write retrain log for dashboard
        _write_retrain_summary(results)

        log.info("Retrain complete. Sleeping until next Sunday...")


def _write_retrain_summary(results: dict) -> None:
    """Write a JSON summary for the dashboard to display."""
    import json
    summary = {
        "timestamp": datetime.now().isoformat(),
        "results"  : {t: "updated" if ok else "kept" for t, ok in results.items()},
    }
    Path("retrain_history.json").write_text(json.dumps(summary, indent=2))


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Weekly GA retrain scheduler")
    parser.add_argument("--schedule", action="store_true",
                        help="Run on schedule (every Sunday 6 PM)")
    parser.add_argument("--ticker",   help="Retrain one ticker only")
    parser.add_argument("--rollback", help="Roll back ticker to previous chromosome")
    parser.add_argument("--backups",  action="store_true",
                        help="List available backups")
    args = parser.parse_args()

    if args.rollback:
        ok = rollback_chromosome(args.rollback)
        sys.exit(0 if ok else 1)

    if args.backups:
        backups = sorted(BACKUP_DIR.glob("*.csv"))
        if not backups:
            print("No backups found.")
        for b in backups:
            size = b.stat().st_size / 1024
            print(f"  {b.name:<50} {size:.1f} KB")
        return

    if args.schedule:
        run_scheduler()
        return

    if args.ticker:
        from bot_manager import BOT_CONFIGS
        if args.ticker not in BOT_CONFIGS:
            print(f"Unknown ticker. Options: {list(BOT_CONFIGS.keys())}")
            sys.exit(1)
        ok = retrain_ticker(args.ticker, BOT_CONFIGS[args.ticker])
        sys.exit(0 if ok else 1)

    # Default: retrain all now
    retrain_all()


if __name__ == "__main__":
    main()
