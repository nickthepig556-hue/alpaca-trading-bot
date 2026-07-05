@echo off
title Alpaca Trading Bot — Launcher
cd /d "C:\Users\nickt\alpaca-trading-bot"
call venv\Scripts\activate

echo Starting API server...
start "API Server" cmd /k "cd /d C:\Users\nickt\alpaca-trading-bot && call venv\Scripts\activate && python api.py"
timeout /t 5 /nobreak >nul

echo Starting bot manager...
start "Bot Manager" cmd /k "cd /d C:\Users\nickt\alpaca-trading-bot && call venv\Scripts\activate && python bot_manager.py"
timeout /t 3 /nobreak >nul

echo Opening dashboard...
start "" "C:\Users\nickt\alpaca-trading-bot\dashboard.html"

echo.
echo ================================================
echo  All components started:
echo    - API Server    (localhost:5000)
echo    - Bot Manager   (9 bots)
echo    - Dashboard     (dashboard.html)
echo ================================================
echo  Close this window when ready.
pause
