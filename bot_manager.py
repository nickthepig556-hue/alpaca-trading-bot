"""
bot_manager.py  —  Step 7: Multi-Bot Manager
=============================================
Runs GLD, SPY, and BTC-USD bots as separate processes.
Each bot has its own GA chromosome, risk limits, and log file.
The manager monitors all processes and restarts any that crash.

Usage
-----
python bot_manager.py              # start all bots
python bot_manager.py --list       # show configured bots
python bot_manager.py --bot GLD    # start only the GLD bot

Requirements
------------
All three chromosomes must exist before starting:
  GLD_best_chromosome.csv
  SPY_best_chromosome.csv
  BTC-USD_best_chromosome.csv

If a chromosome is missing, run:
  python train_bot.py --ticker GLD
  python train_bot.py --ticker SPY
  python train_bot.py --ticker BTC-USD
"""

import os
import sys
import time
import signal
import logging
import argparse
import json
import multiprocessing
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# BOT CONFIGURATIONS  — each bot is fully independent
# ──────────────────────────────────────────────────────────────────────────────

BOT_CONFIGS = {
    "GLD": {
        "id"                 : "b1",
        "name"               : "GLD bot",
        "ticker"             : "GLD",
        "chromosome_file"    : "GLD_best_chromosome.csv",
        "log_file"           : "GLD_bot.log",

        # GA training settings
        "ga": {
            "population_size"    : 100,
            "generations"        : 200,
            "elite_count"        : 5,
            "tournament_size"    : 5,
            "crossover_rate"     : 0.85,
            "mutation_rate_init" : 0.15,
            "mutation_rate_min"  : 0.01,
            "mutation_rate_max"  : 0.30,
            "mutation_step"      : 0.15,
            "stagnation_window"  : 20,
            "fitness_alpha"      : 0.60,
            "fitness_beta"       : 0.40,
            "drawdown_penalty"   : 0.50,
            "max_drawdown_thresh": 0.20,
            "min_trades"         : 10,
            "random_seed"        : 42,
        },

        # Risk / execution settings
        "risk": {
            "min_allocation_pct" : 0.05,
            "max_allocation_pct" : 0.35,   # conservative — gold is less volatile
            "weight_threshold"   : 0.30,
            "stop_loss_pct"      : 0.02,
            "take_profit_pct"    : 0.04,
            "trailing_stop_pct"  : 0.015,
        },

        "market_open_delay_s": 300,
        "intraday_interval_s": 60,
        "lookback_days"      : 60,
        "start_date"         : "2020-01-01",
    },

    "SPY": {
        "id"                 : "b2",
        "name"               : "SPY bot",
        "ticker"             : "SPY",
        "chromosome_file"    : "SPY_best_chromosome.csv",
        "log_file"           : "SPY_bot.log",

        "ga": {
            "population_size"    : 120,    # larger pop — SPY has more noise
            "generations"        : 250,
            "elite_count"        : 6,
            "tournament_size"    : 5,
            "crossover_rate"     : 0.85,
            "mutation_rate_init" : 0.12,
            "mutation_rate_min"  : 0.01,
            "mutation_rate_max"  : 0.25,
            "mutation_step"      : 0.12,
            "stagnation_window"  : 25,
            "fitness_alpha"      : 0.55,
            "fitness_beta"       : 0.45,
            "drawdown_penalty"   : 0.60,   # stricter — SPY drawdowns can be deep
            "max_drawdown_thresh": 0.15,
            "min_trades"         : 12,
            "random_seed"        : 43,
        },

        "risk": {
            "min_allocation_pct" : 0.05,
            "max_allocation_pct" : 0.40,
            "weight_threshold"   : 0.28,
            "stop_loss_pct"      : 0.015,  # tighter stop — SPY moves fast
            "take_profit_pct"    : 0.03,
            "trailing_stop_pct"  : 0.012,
        },

        "market_open_delay_s": 300,
        "intraday_interval_s": 45,         # check more often — higher liquidity
        "lookback_days"      : 60,
        "start_date"         : "2020-01-01",
    },


}

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [manager]  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot_manager.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("manager")


# ──────────────────────────────────────────────────────────────────────────────
# BOT WORKER  — runs in its own process
# ──────────────────────────────────────────────────────────────────────────────

def run_bot_process(config: dict, state_file: str = "bot_state.json") -> None:
    """
    Entry point for each bot subprocess.
    Imports alpaca_bot and overrides its config with per-bot settings,
    then runs the main trading loop.
    """
    ticker = config["ticker"]

    # Set up per-bot logging
    bot_log = logging.getLogger(ticker)
    bot_log.setLevel(logging.INFO)
    fh = logging.FileHandler(config["log_file"], encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s"
    ))
    bot_log.addHandler(fh)
    bot_log.addHandler(logging.StreamHandler())

    bot_log.info(f"[{ticker}] Process started  PID={os.getpid()}")

    # Write PID to state file so manager/dashboard can track it
    _update_state_pid(state_file, config["id"], os.getpid())

    try:
        import alpaca_bot as ab
        import numpy as np
        import pandas as pd

        # Override global config with per-bot settings
        ab.TICKER            = ticker
        ab.CHROMOSOME_FILE   = config["chromosome_file"]
        ab.BOT_CONFIG.update({
            **config["risk"],
            "market_open_delay_s" : config["market_open_delay_s"],
            "intraday_interval_s" : config["intraday_interval_s"],
            "lookback_days"       : config["lookback_days"],
        })

        # Redirect alpaca_bot's logger to the per-bot file
        for h in ab.log.handlers[:]:
            ab.log.removeHandler(h)
        ab.log.addHandler(fh)

        bot_log.info(f"[{ticker}] Config loaded — max alloc "
                     f"{config['risk']['max_allocation_pct']*100:.0f}%  "
                     f"stop {config['risk']['stop_loss_pct']*100:.1f}%")

        # Run the bot loop
        ab.run_bot()

    except FileNotFoundError as e:
        bot_log.error(f"[{ticker}] Chromosome not found: {e}")
        bot_log.error(f"[{ticker}] Run: python train_bot.py --ticker {ticker}")
    except Exception as e:
        bot_log.error(f"[{ticker}] Crashed: {e}", exc_info=True)
    finally:
        _update_state_pid(state_file, config["id"], None)
        bot_log.info(f"[{ticker}] Process exiting")


def _update_state_pid(state_file: str, bot_id: str, pid) -> None:
    """Update the PID field in bot_state.json for this bot."""
    try:
        p = Path(state_file)
        bots = json.loads(p.read_text()) if p.exists() else []
        for b in bots:
            if b["id"] == bot_id:
                b["pid"] = pid
                break
        p.write_text(json.dumps(bots, indent=2))
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# MANAGER
# ──────────────────────────────────────────────────────────────────────────────

class BotManager:
    """
    Launches and supervises one process per bot.
    Restarts crashed bots after a cooldown period.
    Handles Ctrl+C by shutting all bots down cleanly.
    """

    RESTART_COOLDOWN_S = 60     # wait 60s before restarting a crashed bot
    MAX_RESTARTS       = 5      # give up after this many crashes in a session

    def __init__(self, tickers: list[str]):
        self.tickers   = tickers
        self.processes : dict[str, multiprocessing.Process] = {}
        self.restarts  : dict[str, int] = {t: 0 for t in tickers}
        self.running   = True

        signal.signal(signal.SIGINT,  self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        log.info("Shutdown signal received — stopping all bots...")
        self.running = False
        self.stop_all()
        sys.exit(0)

    def check_chromosomes(self) -> bool:
        """Warn about any missing chromosome files before launching."""
        ok = True
        for ticker in self.tickers:
            cfg  = BOT_CONFIGS[ticker]
            path = Path(cfg["chromosome_file"])
            if not path.exists():
                log.warning(
                    f"[{ticker}] Chromosome missing: {path}  "
                    f"→ run: python train_bot.py --ticker {ticker}"
                )
                ok = False
            else:
                log.info(f"[{ticker}] Chromosome found: {path}")
        return ok

    def _launch(self, ticker: str) -> None:
        cfg = BOT_CONFIGS[ticker]
        p   = multiprocessing.Process(
            target=run_bot_process,
            args=(cfg,),
            name=f"bot-{ticker}",
            daemon=True,
        )
        p.start()
        self.processes[ticker] = p
        log.info(f"[{ticker}] Launched  PID={p.pid}")

    def start_all(self) -> None:
        log.info("=" * 55)
        log.info("  Multi-Bot Manager starting")
        log.info(f"  Bots: {', '.join(self.tickers)}")
        log.info("=" * 55)

        for ticker in self.tickers:
            self._launch(ticker)
            time.sleep(2)   # stagger launches to avoid API rate limits

    def stop_all(self) -> None:
        for ticker, p in self.processes.items():
            if p.is_alive():
                log.info(f"[{ticker}] Stopping PID={p.pid}...")
                p.terminate()
                p.join(timeout=10)
                if p.is_alive():
                    p.kill()
                log.info(f"[{ticker}] Stopped")

    def supervise(self) -> None:
        """
        Main supervision loop — checks every 30s whether any bot has crashed
        and restarts it if under the restart limit.
        """
        log.info("Supervision loop running (Ctrl+C to stop all bots)")

        while self.running:
            time.sleep(30)
# Auto-detect new bots from bot_configs.json
            try:
                from ticker_manager import load_all_configs
                dynamic = load_all_configs()
                for cfg in dynamic.values():
                    ticker = cfg["ticker"]
                    if (cfg.get("status") == "running" and
                        ticker not in self.processes and
                        ticker not in self.tickers and
                        Path(cfg["chromosome_file"]).exists()):
                        log.info(f"[{ticker}] New bot detected — launching automatically")
                        BOT_CONFIGS[ticker] = cfg
                        self.tickers.append(ticker)
                        self.restarts[ticker] = 0
                        self._launch(ticker)
            except Exception as e:
                log.warning(f"Auto-detect error: {e}")


            for ticker in self.tickers:
                p = self.processes.get(ticker)
                if p is None:
                    continue

                if not p.is_alive():
                    exit_code = p.exitcode
                    restarts  = self.restarts[ticker]

                    if restarts >= self.MAX_RESTARTS:
                        log.error(
                            f"[{ticker}] Reached max restarts ({self.MAX_RESTARTS}). "
                            f"Not restarting. Check {BOT_CONFIGS[ticker]['log_file']}"
                        )
                        continue

                    log.warning(
                        f"[{ticker}] Process exited (code={exit_code}). "
                        f"Restart {restarts + 1}/{self.MAX_RESTARTS} "
                        f"in {self.RESTART_COOLDOWN_S}s..."
                    )
                    time.sleep(self.RESTART_COOLDOWN_S)

                    self.restarts[ticker] += 1
                    self._launch(ticker)

            # Print status summary every 5 minutes
            if int(time.time()) % 300 < 30:
                self._print_status()

    def _print_status(self) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        log.info(f"--- Status at {now} ---")
        for ticker in self.tickers:
            p = self.processes.get(ticker)
            alive = p.is_alive() if p else False
            pid   = p.pid if p and alive else "—"
            log.info(
                f"  {ticker:<10} {'[RUNNING]' if alive else '[STOPPED]'}"
                f"  PID={pid}  restarts={self.restarts[ticker]}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING HELPER  — generates chromosomes for all bots
# ──────────────────────────────────────────────────────────────────────────────

def train_all(tickers: list[str]) -> None:
    """
    Run the GA training pipeline for each ticker that is missing a chromosome.
    Runs sequentially (training is CPU-heavy; no point parallelising).
    """
    import pandas as pd
    from stock_data import download_stock_data, add_indicators, preprocess_data
    from genetic_algorithm import run_ga, save_results, FEATURE_COLS

    for ticker in tickers:
        cfg  = BOT_CONFIGS[ticker]
        path = Path(cfg["chromosome_file"])

        if path.exists():
            log.info(f"[{ticker}] Chromosome already exists — skipping training")
            continue

        log.info(f"[{ticker}] Training GA ({cfg['ga']['generations']} generations)...")

        try:
            df_raw    = download_stock_data(ticker, cfg["start_date"],
                                            datetime.now().strftime("%Y-%m-%d"))
            df_ind    = add_indicators(df_raw.copy())
            df_scaled, _ = preprocess_data(df_ind)

            # Align indices
            idx       = df_scaled.index.intersection(df_raw.index)
            df_scaled = df_scaled.loc[idx]
            df_raw_al = df_raw.loc[idx]

            result = run_ga(df_scaled, df_raw_al, cfg["ga"])
            save_results(result, ticker)
            log.info(f"[{ticker}] Training complete — fitness {result['best_fitness']:.4f}")

        except Exception as e:
            log.error(f"[{ticker}] Training failed: {e}", exc_info=True)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-bot manager")
    parser.add_argument("--bot",   help="Run only this ticker (e.g. GLD)")
    parser.add_argument("--train", action="store_true",
                        help="Train missing chromosomes before starting")
    parser.add_argument("--list",  action="store_true",
                        help="List all configured bots and exit")
    args = parser.parse_args()

    # Load dynamic configs and merge into BOT_CONFIGS
    try:
        from pathlib import Path as _Path
        from ticker_manager import load_all_configs
        dynamic = load_all_configs()
        for cfg in dynamic.values():
            ticker = cfg["ticker"]
            if cfg["status"] == "running" and _Path(cfg["chromosome_file"]).exists():
                BOT_CONFIGS[ticker] = cfg
    except Exception as e:
        log.warning(f"Could not load dynamic configs: {e}")

    tickers = [args.bot] if args.bot else list(BOT_CONFIGS.keys())

    # Validate ticker names
    for t in tickers:
        if t not in BOT_CONFIGS:
            print(f"Unknown ticker: {t}. Valid options: {list(BOT_CONFIGS.keys())}")
            sys.exit(1)

    if args.list:
        print("\nConfigured bots:")
        for t, cfg in BOT_CONFIGS.items():
            exists = "[OK]" if Path(cfg["chromosome_file"]).exists() else "[MISSING CHROMOSOME]"
            print(f"  {t:<12} {exists}  alloc<={cfg['risk']['max_allocation_pct']*100:.0f}%"
                  f"  stop={cfg['risk']['stop_loss_pct']*100:.1f}%"
                  f"  {cfg['ga']['generations']}gen")
        print()
        return

    if args.train:
        train_all(tickers)

    manager = BotManager(tickers)
    manager.check_chromosomes()
    manager.start_all()
    manager.supervise()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
