@echo off
title Alpaca Bot Manager
cd /d "C:\Users\nickt\alpaca-trading-bot"
call venv\Scripts\activate
python bot_manager.py
pause
