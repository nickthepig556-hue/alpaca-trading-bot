@echo off
title Stopping all bots
echo Stopping all Alpaca bot processes...
taskkill /FI "WINDOWTITLE eq API Server*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Bot Manager*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Alpaca*" /F >nul 2>&1
echo All processes stopped.
pause
