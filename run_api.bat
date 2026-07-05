@echo off
title Alpaca API Server
cd /d "C:\Users\nickt\alpaca-trading-bot"
call venv\Scripts\activate
python api.py
pause
