"""
bot_manager_dynamic_patch.py  —  CHANGES TO MAKE IN bot_manager.py
=====================================================================
This makes bot_manager.py read bots from bot_configs.json (created via
the dashboard) INSTEAD OF (in addition to) the hardcoded BOT_CONFIGS
dict, so any ticker created through the UI gets picked up automatically.

────────────────────────────────────────────────────────────────────
CHANGE 1 — Add these imports near the top of bot_manager.py:
────────────────────────────────────────────────────────────────────

from pathlib import Path
from ticker_manager import load_all_configs, update_bot_status


────────────────────────────────────────────────────────────────────
CHANGE 2 — In main(), find this line:
────────────────────────────────────────────────────────────────────

    tickers = [args.bot] if args.bot else list(BOT_CONFIGS.keys())

REPLACE WITH:

    dynamic_configs = load_all_configs()
    runnable = {
        cfg["ticker"]: cfg for cfg in dynamic_configs.values()
        if cfg["status"] in ("running", "training") and
           Path(cfg["chromosome_file"]).exists()
    }
    if args.bot:
        tickers = [args.bot]
    else:
        tickers = list(runnable.keys())

    for ticker, cfg in runnable.items():
        BOT_CONFIGS[ticker] = {
            "id"                  : cfg["id"],
            "name"                : cfg["name"],
            "ticker"              : cfg["ticker"],
            "chromosome_file"     : cfg["chromosome_file"],
            "log_file"            : cfg["log_file"],
            "ga"                  : cfg["ga"],
            "risk"                : cfg["risk"],
            "market_open_delay_s" : cfg["market_open_delay_s"],
            "intraday_interval_s" : cfg["intraday_interval_s"],
            "lookback_days"       : cfg["lookback_days"],
            "start_date"          : cfg["start_date"],
        }


────────────────────────────────────────────────────────────────────
CHANGE 3 — In the supervise() method, find this line:
────────────────────────────────────────────────────────────────────

            time.sleep(30)

The very next line after it — add this new bot detection block:

            dynamic_configs = load_all_configs()
            for cfg in dynamic_configs.values():
                ticker = cfg["ticker"]
                if (cfg["status"] == "running" and
                    ticker not in self.processes and
                    Path(cfg["chromosome_file"]).exists()):
                    log.info(f"[{ticker}] New bot detected -- launching")
                    BOT_CONFIGS[ticker] = {
                        "id": cfg["id"], "name": cfg["name"], "ticker": ticker,
                        "chromosome_file": cfg["chromosome_file"],
                        "log_file": cfg["log_file"], "ga": cfg["ga"],
                        "risk": cfg["risk"],
                        "market_open_delay_s": cfg["market_open_delay_s"],
                        "intraday_interval_s": cfg["intraday_interval_s"],
                        "lookback_days": cfg["lookback_days"],
                        "start_date": cfg["start_date"],
                    }
                    self.tickers.append(ticker)
                    self.restarts[ticker] = 0
                    self._launch(ticker)

So the full block looks like:

    def supervise(self) -> None:
        log.info("Supervision loop running (Ctrl+C to stop all bots)")

        while self.running:
            time.sleep(30)

            dynamic_configs = load_all_configs()
            for cfg in dynamic_configs.values():
                ticker = cfg["ticker"]
                if (cfg["status"] == "running" and
                    ticker not in self.processes and
                    Path(cfg["chromosome_file"]).exists()):
                    log.info(f"[{ticker}] New bot detected -- launching")
                    BOT_CONFIGS[ticker] = {
                        "id": cfg["id"], "name": cfg["name"], "ticker": ticker,
                        "chromosome_file": cfg["chromosome_file"],
                        "log_file": cfg["log_file"], "ga": cfg["ga"],
                        "risk": cfg["risk"],
                        "market_open_delay_s": cfg["market_open_delay_s"],
                        "intraday_interval_s": cfg["intraday_interval_s"],
                        "lookback_days": cfg["lookback_days"],
                        "start_date": cfg["start_date"],
                    }
                    self.tickers.append(ticker)
                    self.restarts[ticker] = 0
                    self._launch(ticker)

            for ticker in self.tickers:
                p = self.processes.get(ticker)
                ... (rest of existing code unchanged)


This means: once you create a bot in the dashboard and it finishes
training, the running bot_manager.py process detects it within 30
seconds and launches it automatically -- no restart needed.
"""

print(__doc__)
