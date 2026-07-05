@echo off
cd /d "C:\Users\nickt\alpaca-trading-bot"
call venv\Scripts\activate
python deploy.py --rotate
