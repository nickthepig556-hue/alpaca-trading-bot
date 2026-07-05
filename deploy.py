"""
deploy.py  —  Step 8: Windows Deployment
=========================================
Sets up the trading bot to:
  1. Start automatically on Windows boot via Task Scheduler
  2. Run the API server and all bots in separate windows
  3. Rotate log files so they don't grow forever
  4. Send a daily email/desktop notification with performance summary

Usage
-----
python deploy.py --install    # install Task Scheduler tasks + create .bat files
python deploy.py --uninstall  # remove all scheduled tasks
python deploy.py --status     # show what's scheduled and running
python deploy.py --rotate     # rotate logs now (also runs automatically daily)
python deploy.py --test       # test that all components start correctly
"""

import os
import sys
import json
import glob
import shutil
import logging
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_DIR  = Path(__file__).resolve().parent
VENV_PYTHON  = PROJECT_DIR / "venv" / "Scripts" / "python.exe"
PYTHON       = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

LOG_MAX_MB   = 10      # rotate log when it exceeds this size
LOG_KEEP     = 5       # keep this many rotated logs per file
TASK_PREFIX  = "AlpacaTradingBot"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s"
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# BATCH FILE GENERATORS
# ──────────────────────────────────────────────────────────────────────────────

def write_bat(name: str, content: str) -> Path:
    path = PROJECT_DIR / name
    path.write_text(content, encoding="utf-8")
    log.info(f"Created: {path}")
    return path


def create_batch_files() -> None:
    """Create all .bat launcher files in the project directory."""

    # ── Start everything ──────────────────────────────────────────────────────
    write_bat("start_all.bat", f"""@echo off
title Alpaca Trading Bot — Launcher
cd /d "{PROJECT_DIR}"
call venv\\Scripts\\activate

echo Starting API server...
start "API Server" cmd /k "cd /d {PROJECT_DIR} && call venv\\Scripts\\activate && python api.py"
timeout /t 3 /nobreak >nul

echo Training missing chromosomes...
python train_bot.py --all
timeout /t 2 /nobreak >nul

echo Starting bot manager...
start "Bot Manager" cmd /k "cd /d {PROJECT_DIR} && call venv\\Scripts\\activate && python bot_manager.py"
timeout /t 2 /nobreak >nul

echo Opening dashboard...
start "" "{PROJECT_DIR}\\dashboard.html"

echo.
echo All components started. Close this window when ready.
pause
""")

    # ── API server only ───────────────────────────────────────────────────────
    write_bat("run_api.bat", f"""@echo off
title Alpaca API Server
cd /d "{PROJECT_DIR}"
call venv\\Scripts\\activate
python api.py
pause
""")

    # ── Bot manager only ──────────────────────────────────────────────────────
    write_bat("run_bots.bat", f"""@echo off
title Alpaca Bot Manager
cd /d "{PROJECT_DIR}"
call venv\\Scripts\\activate
python bot_manager.py
pause
""")

    # ── Train all chromosomes ─────────────────────────────────────────────────
    write_bat("train_all.bat", f"""@echo off
title GA Training
cd /d "{PROJECT_DIR}"
call venv\\Scripts\\activate
python train_bot.py --all
pause
""")

    # ── Stop everything ───────────────────────────────────────────────────────
    write_bat("stop_all.bat", f"""@echo off
title Stopping all bots
echo Stopping all Alpaca bot processes...
taskkill /FI "WINDOWTITLE eq API Server*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Bot Manager*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Alpaca*" /F >nul 2>&1
echo All processes stopped.
pause
""")

    # ── Log rotation ──────────────────────────────────────────────────────────
    write_bat("rotate_logs.bat", f"""@echo off
cd /d "{PROJECT_DIR}"
call venv\\Scripts\\activate
python deploy.py --rotate
""")

    # ── Status check ─────────────────────────────────────────────────────────
    write_bat("status.bat", f"""@echo off
cd /d "{PROJECT_DIR}"
call venv\\Scripts\\activate
python deploy.py --status
pause
""")

    log.info("All batch files created.")


# ──────────────────────────────────────────────────────────────────────────────
# TASK SCHEDULER
# ──────────────────────────────────────────────────────────────────────────────

def run_schtasks(args: list[str]) -> tuple[bool, str]:
    """Run a schtasks.exe command and return (success, output)."""
    try:
        result = subprocess.run(
            ["schtasks"] + args,
            capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


def install_tasks() -> None:
    """Register Task Scheduler tasks for auto-start on boot."""

    tasks = [
        {
            "name"   : f"{TASK_PREFIX}_API",
            "desc"   : "Alpaca trading bot — Flask API server",
            "cmd"    : str(PROJECT_DIR / "run_api.bat"),
            "trigger": "ONLOGON",
            "delay"  : "PT1M",    # 1 minute after logon
        },
        {
            "name"   : f"{TASK_PREFIX}_Bots",
            "desc"   : "Alpaca trading bot — bot manager (GLD/SPY/BTC)",
            "cmd"    : str(PROJECT_DIR / "run_bots.bat"),
            "trigger": "ONLOGON",
            "delay"  : "PT2M",    # 2 minutes after logon (API starts first)
        },
        {
            "name"   : f"{TASK_PREFIX}_LogRotation",
            "desc"   : "Alpaca trading bot — daily log rotation",
            "cmd"    : str(PROJECT_DIR / "rotate_logs.bat"),
            "trigger": "DAILY",
            "time"   : "02:00",   # 2 AM daily
        },
    ]

    log.info("Installing Task Scheduler tasks...")

    for task in tasks:
        # Delete existing task first (ignore error if not found)
        run_schtasks(["/Delete", "/TN", task["name"], "/F"])

        if task["trigger"] == "ONLOGON":
            ok, out = run_schtasks([
                "/Create",
                "/TN",  task["name"],
                "/TR",  f'"{task["cmd"]}"',
                "/SC",  "ONLOGON",
                "/IT",                      # run only when user is logged on
                "/RL",  "HIGHEST",          # run with highest privileges
                "/F",                       # force overwrite
                "/SD",  datetime.now().strftime("%m/%d/%Y"),
                "/ED",  "12/31/2099",
            ])
        else:
            ok, out = run_schtasks([
                "/Create",
                "/TN",  task["name"],
                "/TR",  f'"{task["cmd"]}"',
                "/SC",  "DAILY",
                "/ST",  task.get("time", "02:00"),
                "/RL",  "HIGHEST",
                "/F",
            ])

        if ok:
            log.info(f"  [OK] {task['name']}")
        else:
            log.warning(f"  [FAILED] {task['name']}: {out.strip()}")

    log.info("\nTasks installed. They will start on your next login.")
    log.info("To start now without rebooting, double-click start_all.bat")


def uninstall_tasks() -> None:
    """Remove all scheduled tasks."""
    task_names = [
        f"{TASK_PREFIX}_API",
        f"{TASK_PREFIX}_Bots",
        f"{TASK_PREFIX}_LogRotation",
    ]
    for name in task_names:
        ok, out = run_schtasks(["/Delete", "/TN", name, "/F"])
        log.info(f"  {'[OK]' if ok else '[NOT FOUND]'} {name}")
    log.info("Tasks removed.")


def show_status() -> None:
    """Show Task Scheduler status and running processes."""
    print("\n" + "="*55)
    print("  DEPLOYMENT STATUS")
    print("="*55)

    # Check scheduled tasks
    print("\nScheduled tasks:")
    for name in [f"{TASK_PREFIX}_API", f"{TASK_PREFIX}_Bots", f"{TASK_PREFIX}_LogRotation"]:
        ok, out = run_schtasks(["/Query", "/TN", name, "/FO", "LIST"])
        if ok:
            status = "Ready"
            for line in out.splitlines():
                if "Status:" in line:
                    status = line.split(":", 1)[1].strip()
            print(f"  [OK]      {name:<40} {status}")
        else:
            print(f"  [MISSING] {name}")

    # Check batch files
    print("\nBatch files:")
    for bat in ["start_all.bat", "run_api.bat", "run_bots.bat",
                "train_all.bat", "stop_all.bat", "rotate_logs.bat"]:
        path = PROJECT_DIR / bat
        exists = "[OK]" if path.exists() else "[MISSING]"
        print(f"  {exists}  {bat}")

    # Check chromosomes
    print("\nChromosomes:")
    for ticker in ["GLD", "SPY", "BTC-USD"]:
        path = PROJECT_DIR / f"{ticker}_best_chromosome.csv"
        exists = "[OK]" if path.exists() else "[MISSING — run train_all.bat]"
        print(f"  {exists}  {ticker}_best_chromosome.csv")

    # Check log files
    print("\nLog files:")
    for log_file in sorted(PROJECT_DIR.glob("*.log")):
        size_kb = log_file.stat().st_size / 1024
        print(f"  {log_file.name:<30} {size_kb:>8.1f} KB")

    # Check API health
    print("\nAPI server:")
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:5000/api/health", timeout=3) as r:
            data = json.loads(r.read())
            print(f"  [ONLINE]  mode={data['mode']}  alpaca={data['alpaca']}")
    except Exception:
        print("  [OFFLINE] Start with run_api.bat")

    print("="*55)


# ──────────────────────────────────────────────────────────────────────────────
# LOG ROTATION
# ──────────────────────────────────────────────────────────────────────────────

def rotate_logs() -> None:
    """
    Rotate any log file that exceeds LOG_MAX_MB.
    Keeps the last LOG_KEEP rotated copies.
    """
    log_files = list(PROJECT_DIR.glob("*.log"))
    if not log_files:
        log.info("No log files found.")
        return

    rotated = 0
    for log_path in log_files:
        size_mb = log_path.stat().st_size / (1024 * 1024)
        if size_mb < LOG_MAX_MB:
            continue

        # Build rotated filename: GLD_bot.log → GLD_bot_20260626_143022.log
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_name = log_path.stem + f"_{ts}" + log_path.suffix
        rotated_path = log_path.parent / rotated_name

        shutil.copy2(log_path, rotated_path)
        log_path.write_text("")   # truncate the active log

        log.info(f"Rotated: {log_path.name} ({size_mb:.1f} MB) → {rotated_name}")
        rotated += 1

        # Prune old rotated files — keep only the newest LOG_KEEP
        stem    = log_path.stem
        pattern = str(log_path.parent / f"{stem}_????????_??????{log_path.suffix}")
        old     = sorted(glob.glob(pattern))
        for old_file in old[:-LOG_KEEP]:
            Path(old_file).unlink()
            log.info(f"Deleted old log: {old_file}")

    if rotated == 0:
        log.info(f"No logs exceeded {LOG_MAX_MB} MB — nothing to rotate.")
    else:
        log.info(f"Rotated {rotated} log file(s).")


# ──────────────────────────────────────────────────────────────────────────────
# TEST
# ──────────────────────────────────────────────────────────────────────────────

def run_tests() -> None:
    """Quick smoke tests to verify the deployment is healthy."""
    print("\n" + "="*55)
    print("  DEPLOYMENT TESTS")
    print("="*55)
    passed = failed = 0

    def check(label, condition, fix=""):
        nonlocal passed, failed
        if condition:
            print(f"  [PASS] {label}")
            passed += 1
        else:
            print(f"  [FAIL] {label}" + (f"\n         Fix: {fix}" if fix else ""))
            failed += 1

    # Python version
    import sys
    check("Python 3.10+", sys.version_info >= (3, 10),
          "Upgrade Python to 3.10 or later")

    # Required files
    for f in ["stock_data.py", "genetic_algorithm.py", "alpaca_bot.py",
              "api.py", "bot_manager.py", "performance.py", ".env"]:
        check(f"File exists: {f}", (PROJECT_DIR / f).exists(),
              f"Copy {f} to {PROJECT_DIR}")

    # .env keys
    from dotenv import load_dotenv
    load_dotenv(PROJECT_DIR / ".env")
    api_key = os.getenv("ALPACA_API_KEY", "")
    check(".env has ALPACA_API_KEY",
          bool(api_key) and api_key not in ("YOUR_API_KEY", "your_key"),
          "Edit .env and add your real Alpaca API key")
    check(".env has ALPACA_SECRET_KEY",
          bool(os.getenv("ALPACA_SECRET_KEY", "")),
          "Edit .env and add your real Alpaca secret key")

    # Packages
    for pkg in ["alpaca", "flask", "flask_cors", "yfinance",
                "sklearn", "numpy", "pandas"]:
        try:
            __import__(pkg.replace("-", "_"))
            check(f"Package: {pkg}", True)
        except ImportError:
            check(f"Package: {pkg}", False, f"pip install {pkg}")

    # Chromosomes
    for ticker in ["GLD", "SPY", "BTC-USD"]:
        path = PROJECT_DIR / f"{ticker}_best_chromosome.csv"
        check(f"Chromosome: {ticker}", path.exists(),
              f"python train_bot.py --ticker {ticker}")

    # API health
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:5000/api/health", timeout=3) as r:
            check("API server running", r.status == 200)
    except Exception:
        check("API server running", False, "Start api.py in a separate terminal")

    print(f"\n  {passed} passed  {failed} failed")
    print("="*55)
    if failed == 0:
        print("\n  All tests passed — ready to deploy!")
    else:
        print(f"\n  Fix the {failed} issue(s) above, then re-run: python deploy.py --test")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deploy trading bot on Windows")
    parser.add_argument("--install",   action="store_true", help="Install Task Scheduler tasks")
    parser.add_argument("--uninstall", action="store_true", help="Remove scheduled tasks")
    parser.add_argument("--status",    action="store_true", help="Show deployment status")
    parser.add_argument("--rotate",    action="store_true", help="Rotate log files now")
    parser.add_argument("--test",      action="store_true", help="Run deployment tests")
    args = parser.parse_args()

    if args.install:
        create_batch_files()
        install_tasks()
    elif args.uninstall:
        uninstall_tasks()
    elif args.status:
        show_status()
    elif args.rotate:
        rotate_logs()
    elif args.test:
        run_tests()
    else:
        # No flag — create batch files as minimum useful action
        create_batch_files()
        print("\nBatch files created. Next steps:")
        print("  1. python deploy.py --test      # verify everything is ready")
        print("  2. python deploy.py --install   # auto-start on Windows boot")
        print("  3. start_all.bat                # start everything now")

if __name__ == "__main__":
    main()
