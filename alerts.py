"""
alerts.py  —  Email Alerts via Gmail
=====================================
Sends email notifications for:
  • Every BUY / SELL order placed
  • Stop-loss and take-profit triggers
  • Daily P&L summary (sent at 4:30 PM ET)
  • Risk warnings (drawdown, loss streak, halt)
  • Bot crash / restart events
  • Weekly retrain completion

Setup (one-time)
----------------
1. Go to https://myaccount.google.com/apppasswords
2. Sign in → Select app: Mail → Select device: Windows Computer
3. Click Generate → copy the 16-character password
4. Add to your .env file:
       ALERT_EMAIL_FROM=your.gmail@gmail.com
       ALERT_EMAIL_TO=your.gmail@gmail.com
       ALERT_EMAIL_PASSWORD=xxxx xxxx xxxx xxxx   (the app password)

Usage
-----
from alerts import Alerter
alerter = Alerter()
alerter.trade_opened("GLD", "long", 10, 231.40, 0.72)
alerter.trade_closed("GLD", "long", 10, 231.40, 235.00, "take-profit")
alerter.daily_summary({"GLD": {...}, "SPY": {...}, "BTC/USD": {...}})
alerter.warning("GLD", "Max drawdown exceeded 20%")
alerter.bot_crashed("SPY", "Connection timeout")
"""

import os
import smtplib
import logging
import threading
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from queue import Queue, Empty

from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

EMAIL_FROM     = os.getenv("ALERT_EMAIL_FROM", "")
EMAIL_TO       = os.getenv("ALERT_EMAIL_TO", "")
EMAIL_PASSWORD = os.getenv("ALERT_EMAIL_PASSWORD", "")
SMTP_HOST      = "smtp.gmail.com"
SMTP_PORT      = 587


# ──────────────────────────────────────────────────────────────────────────────
# HTML EMAIL TEMPLATES
# ──────────────────────────────────────────────────────────────────────────────

STYLE = """
<style>
  body { font-family: system-ui, sans-serif; background: #f4f3ef; margin: 0; padding: 20px; }
  .card { background: #fff; border-radius: 12px; padding: 24px; max-width: 520px;
          margin: 0 auto; border: 1px solid #e1e0d9; }
  .header { font-size: 13px; color: #898781; margin-bottom: 16px; }
  h2 { margin: 0 0 4px; font-size: 22px; font-weight: 500; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 99px;
           font-size: 12px; font-weight: 500; margin-bottom: 16px; }
  .badge-buy   { background: #eaf3de; color: #27500a; }
  .badge-sell  { background: #fcebeb; color: #791f1f; }
  .badge-short { background: #e6f1fb; color: #0c447c; }
  .badge-cover { background: #faeeda; color: #633806; }
  .badge-warn  { background: #faeeda; color: #633806; }
  .badge-crash { background: #fcebeb; color: #791f1f; }
  .badge-info  { background: #f4f3ef; color: #52514e; }
  .row { display: flex; justify-content: space-between; padding: 8px 0;
         border-bottom: 1px solid #f4f3ef; font-size: 14px; }
  .row:last-child { border-bottom: none; }
  .label { color: #898781; }
  .value { font-weight: 500; }
  .up   { color: #3b6d11; }
  .down { color: #a32d2d; }
  .footer { font-size: 11px; color: #898781; margin-top: 16px; text-align: center; }
</style>
"""


def _html_row(label: str, value: str, cls: str = "") -> str:
    return f'<div class="row"><span class="label">{label}</span><span class="value {cls}">{value}</span></div>'


def _wrap(title: str, badge: str, badge_cls: str, body: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!DOCTYPE html><html><head>{STYLE}</head><body>
<div class="card">
  <div class="header">Alpaca Trading Bot &nbsp;·&nbsp; {now}</div>
  <h2>{title}</h2>
  <span class="badge badge-{badge_cls}">{badge}</span>
  {body}
  <div class="footer">This is an automated alert from your trading bot.</div>
</div></body></html>"""


# ──────────────────────────────────────────────────────────────────────────────
# EMAIL SENDER  (async queue — never blocks the trading loop)
# ──────────────────────────────────────────────────────────────────────────────

class Alerter:
    """
    Thread-safe email alerter.
    Emails are queued and sent in a background thread so the trading
    loop is never delayed waiting for SMTP.
    """

    def __init__(self):
        self._queue  = Queue()
        self._active = bool(EMAIL_FROM and EMAIL_TO and EMAIL_PASSWORD)

        if not self._active:
            log.warning(
                "Email alerts disabled — set ALERT_EMAIL_FROM, "
                "ALERT_EMAIL_TO, ALERT_EMAIL_PASSWORD in .env"
            )
        else:
            log.info(f"Email alerts enabled → {EMAIL_TO}")
            self._worker = threading.Thread(
                target=self._send_loop, daemon=True, name="alert-worker"
            )
            self._worker.start()

    # ── Public API ────────────────────────────────────────────────────────────

    def trade_opened(self, ticker: str, side: str, qty: int,
                     price: float, confidence: float) -> None:
        """Call immediately after a BUY or SELL SHORT order is placed."""
        action     = "BUY (Long)" if side == "long" else "SELL SHORT"
        badge_cls  = "buy" if side == "long" else "short"
        value      = qty * price
        subject    = f"[{ticker}] {action} — {qty} shares @ ${price:,.2f}"

        body = (
            _html_row("Ticker",     ticker) +
            _html_row("Action",     action) +
            _html_row("Quantity",   str(qty)) +
            _html_row("Price",      f"${price:,.2f}") +
            _html_row("Value",      f"${value:,.2f}") +
            _html_row("Confidence", f"{confidence:.1%}") +
            _html_row("Time",       datetime.now().strftime("%H:%M:%S"))
        )
        html = _wrap(f"{action} — {ticker}", action, badge_cls, body)
        self._enqueue(subject, html)

    def trade_closed(self, ticker: str, side: str, qty: int,
                     entry: float, exit_price: float, reason: str) -> None:
        """Call when a position is closed for any reason."""
        pnl        = (exit_price - entry) * qty if side == "long" else (entry - exit_price) * qty
        pnl_pct    = (exit_price - entry) / entry * 100 if side == "long" else (entry - exit_price) / entry * 100
        pnl_cls    = "up" if pnl >= 0 else "down"
        pnl_str    = f"{'+'if pnl>=0 else ''}{pnl_pct:.2f}%  (${pnl:+,.2f})"
        action     = "SELL" if side == "long" else "BUY TO COVER"
        badge_cls  = "sell" if side == "long" else "cover"

        emoji      = "🎯" if "profit" in reason else "⛔" if "stop" in reason else "📤"
        subject    = f"[{ticker}] {action} — {pnl_str}  ({reason})"

        body = (
            _html_row("Ticker",      ticker) +
            _html_row("Action",      action) +
            _html_row("Reason",      reason.title()) +
            _html_row("Quantity",    str(qty)) +
            _html_row("Entry price", f"${entry:,.2f}") +
            _html_row("Exit price",  f"${exit_price:,.2f}") +
            _html_row("P&L",         pnl_str, pnl_cls) +
            _html_row("Side",        side.title())
        )
        html = _wrap(f"Position Closed — {ticker}", reason.title(), badge_cls, body)
        self._enqueue(subject, html)

    def daily_summary(self, bot_stats: dict) -> None:
        """
        Send end-of-day summary for all bots.
        bot_stats = {
          "GLD": { "total_return": 0.087, "win_rate": 63.2,
                   "n_trades": 38, "daily_pnl": 312.50,
                   "portfolio_value": 104832 },
          ...
        }
        """
        total_pnl   = sum(s.get("daily_pnl", 0) for s in bot_stats.values())
        total_value = sum(s.get("portfolio_value", 0) for s in bot_stats.values())
        pnl_cls     = "up" if total_pnl >= 0 else "down"
        subject     = (f"Daily Summary — "
                       f"{'+'if total_pnl>=0 else ''}${total_pnl:,.2f}  "
                       f"| Portfolio ${total_value:,.2f}")

        rows = ""
        for ticker, s in bot_stats.items():
            pnl     = s.get("daily_pnl", 0)
            ret     = s.get("total_return", 0) * 100
            wr      = s.get("win_rate", 0)
            trades  = s.get("n_trades", 0)
            cls     = "up" if pnl >= 0 else "down"
            rows += f"<div style='margin:12px 0;padding:12px;background:#f4f3ef;border-radius:8px'>"
            rows += f"<strong>{ticker}</strong><br>"
            rows += _html_row("Daily P&L",    f"{'+'if pnl>=0 else ''}${pnl:,.2f}", cls)
            rows += _html_row("Total return", f"{'+'if ret>=0 else ''}{ret:.2f}%", "up" if ret>=0 else "down")
            rows += _html_row("Win rate",     f"{wr:.1f}%")
            rows += _html_row("Trades today", str(trades))
            rows += "</div>"

        summary_row = _html_row("Total daily P&L",
                                f"{'+'if total_pnl>=0 else ''}${total_pnl:,.2f}",
                                pnl_cls)
        body = summary_row + "<br>" + rows
        html = _wrap("Daily Summary", "End of Day", "info", body)
        self._enqueue(subject, html)

    def warning(self, ticker: str, message: str) -> None:
        """Risk warning — drawdown exceeded, loss streak, halt triggered."""
        subject = f"[WARNING] {ticker} — {message}"
        body    = (
            _html_row("Bot",     ticker) +
            _html_row("Warning", message) +
            _html_row("Time",    datetime.now().strftime("%H:%M:%S")) +
            _html_row("Action",  "Review bot performance")
        )
        html = _wrap(f"Risk Warning — {ticker}", "Warning", "warn", body)
        self._enqueue(subject, html)

    def bot_crashed(self, ticker: str, error: str) -> None:
        """Bot process crashed or restarted."""
        subject = f"[CRASH] {ticker} bot crashed — restarting"
        body    = (
            _html_row("Bot",   ticker) +
            _html_row("Error", error[:200]) +
            _html_row("Time",  datetime.now().strftime("%H:%M:%S")) +
            _html_row("Action", "Bot will auto-restart in 60s")
        )
        html = _wrap(f"Bot Crashed — {ticker}", "Crashed", "crash", body)
        self._enqueue(subject, html)

    def retrain_complete(self, results: dict) -> None:
        """Weekly retrain finished."""
        subject = "Weekly Retrain Complete"
        rows = ""
        for ticker, status in results.items():
            cls  = "up" if status == "updated" else ""
            rows += _html_row(ticker, status.title(), cls)
        html = _wrap("Weekly Retrain", "Retrain", "info", rows)
        self._enqueue(subject, html)

    def test(self) -> bool:
        """Send a test email to verify configuration. Returns True if sent."""
        subject = "Alpaca Bot — Email Alerts Working"
        body    = (
            _html_row("Status",  "Connected") +
            _html_row("From",    EMAIL_FROM) +
            _html_row("To",      EMAIL_TO) +
            _html_row("Time",    datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        html = _wrap("Test Alert", "Test", "info", body)
        return self._send_now(subject, html)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _enqueue(self, subject: str, html: str) -> None:
        if self._active:
            self._queue.put((subject, html))

    def _send_loop(self) -> None:
        """Background thread — drains the queue and sends emails."""
        while True:
            try:
                subject, html = self._queue.get(timeout=5)
                self._send_now(subject, html)
            except Empty:
                continue
            except Exception as e:
                log.error(f"Alert worker error: {e}")

    def _send_now(self, subject: str, html: str) -> bool:
        """Send immediately — called from background thread."""
        if not self._active:
            log.info(f"[Alert skipped — not configured] {subject}")
            return False
        try:
            msg                    = MIMEMultipart("alternative")
            msg["Subject"]         = subject
            msg["From"]            = EMAIL_FROM
            msg["To"]              = EMAIL_TO
            msg.attach(MIMEText(html, "html"))

            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
                server.starttls()
                server.login(EMAIL_FROM, EMAIL_PASSWORD)
                server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())

            log.info(f"Alert sent: {subject}")
            return True
        except smtplib.SMTPAuthenticationError:
            log.error(
                "Gmail authentication failed. "
                "Make sure you're using an App Password, not your regular password. "
                "Go to https://myaccount.google.com/apppasswords"
            )
            return False
        except Exception as e:
            log.error(f"Failed to send alert: {e}")
            return False


# ──────────────────────────────────────────────────────────────────────────────
# DAILY SUMMARY SCHEDULER
# ──────────────────────────────────────────────────────────────────────────────

def schedule_daily_summary(alerter: "Alerter") -> None:
    """
    Runs in a background thread.
    Waits until 4:30 PM ET then triggers the daily summary email.
    Call this once at bot startup.
    """
    import time
    from datetime import timezone

    def _loop():
        while True:
            try:
                from zoneinfo import ZoneInfo
                now = datetime.now(ZoneInfo("America/New_York"))
            except ImportError:
                now = datetime.utcnow().replace(
                    tzinfo=timezone.utc
                ).astimezone()

            # Target: 4:30 PM ET weekdays
            target = now.replace(hour=16, minute=30, second=0, microsecond=0)
            if now >= target or now.weekday() >= 5:
                # Already past 4:30 or weekend — sleep until tomorrow 4:30
                from datetime import timedelta
                target += timedelta(days=1)
                while target.weekday() >= 5:
                    target += timedelta(days=1)

            wait_s = (target - now).total_seconds()
            log.info(f"Daily summary scheduled for {target.strftime('%Y-%m-%d %H:%M ET')}")
            time.sleep(wait_s)

            # Fetch live stats from API
            try:
                import urllib.request, json
                with urllib.request.urlopen(
                    "http://localhost:5000/api/account", timeout=5
                ) as r:
                    account = json.loads(r.read())

                with urllib.request.urlopen(
                    "http://localhost:5000/api/bots", timeout=5
                ) as r:
                    bots = json.loads(r.read())

                bot_stats = {}
                for b in bots:
                    bot_stats[b["ticker"]] = {
                        "daily_pnl"      : account.get("pnl_today", 0) / max(len(bots), 1),
                        "portfolio_value": account.get("portfolio_value", 0) / max(len(bots), 1),
                        "total_return"   : 0,
                        "win_rate"       : b.get("win_rate", 0),
                        "n_trades"       : b.get("trades", 0),
                    }
                alerter.daily_summary(bot_stats)

            except Exception as e:
                log.warning(f"Could not fetch stats for daily summary: {e}")
                alerter.daily_summary({})

    t = threading.Thread(target=_loop, daemon=True, name="daily-summary")
    t.start()


# ──────────────────────────────────────────────────────────────────────────────
# CLI — test your setup
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nAlpaca Bot — Email Alert Setup")
    print("="*40)
    print(f"From    : {EMAIL_FROM or 'NOT SET'}")
    print(f"To      : {EMAIL_TO   or 'NOT SET'}")
    print(f"Password: {'SET' if EMAIL_PASSWORD else 'NOT SET'}")
    print()

    if not EMAIL_FROM or not EMAIL_PASSWORD:
        print("Add these to your .env file:")
        print("  ALERT_EMAIL_FROM=your.gmail@gmail.com")
        print("  ALERT_EMAIL_TO=your.gmail@gmail.com")
        print("  ALERT_EMAIL_PASSWORD=xxxx xxxx xxxx xxxx")
        print()
        print("Get your app password at:")
        print("  https://myaccount.google.com/apppasswords")
    else:
        print("Sending test email...")
        alerter = Alerter()
        import time; time.sleep(1)
        ok = alerter.test()
        if ok:
            print(f"Test email sent to {EMAIL_TO} — check your inbox!")
        else:
            print("Failed — check the error above.")
