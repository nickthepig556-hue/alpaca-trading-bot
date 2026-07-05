@echo off
title GA Training
cd /d "C:\Users\nickt\alpaca-trading-bot"
call venv\Scripts\activate
python train_bot.py --all
pause
