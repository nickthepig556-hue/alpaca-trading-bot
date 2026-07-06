"""
futures_trading.py  —  Live Futures Trading Loop
=================================================
Extends alpaca_bot.py with futures trading capabilities.
Uses the evolved chromosome from futures_bot.py for signals AND leverage.

Usage
-----
python futures_trading.py --ticker BTC/USD --paper
python futures_trading.py --ticker ETH/USD --paper
python futures_trading.py --list

The bot_manager.py picks this up automatically for any ticker
in FUTURES_CONFIGS once the chromosome is trained.
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Import from existing modules
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

from stock_data import add_indicators, preprocess_data
from futures_bot import (
    FUTURES_CONFIGS, load_futures_chromosome,
    decode_futures_chromosome, FuturesRiskGuard,
)
from alerts import Alerter

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

API_KEY      = os.getenv("ALPACA_API_KEY", "")
SECRET_KEY   = os.getenv("ALPACA_SECRET_KEY", "")
IS_PAPER     = os.getenv("TRADING_MODE", "paper").lower() != "live"
FEATURES     = ['Close','Volume','SMA_20','SMA_50','SMA_200','RSI',
                'MACD','Signal','BB_Upper','BB_Lower','Daily_Return','Volume_Change']

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)
alerter = Alerter()


# ──────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ──────────────────────────────────────────────────────────────────────────────

def fetch_futures_bars(ticker: str, lookback_days: int) -> pd.DataFrame:
    """Fetch bars for futures — crypto via Alpaca, index/commodity via yfinance."""
    cfg = FUTURES_CONFIGS[ticker]

    if cfg["type"] == "crypto":
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest
        import pandas as _pd

        end   = _pd.Timestamp.now(tz='America/New_York')
        start = end - _pd.Timedelta(days=lookback_days + 10)

        client  = CryptoHistoricalDataClient()
        request = CryptoBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start.isoformat(),
            end=end.isoformat(),
        )
        bars = client.get_crypto_bars(request).df
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(ticker, level='symbol')
        bars.index = pd.to_datetime(bars.index).tz_localize(None)
        bars.index.name = 'Date'
        bars = bars.rename(columns={
            'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'
        })[['Open','High','Low','Close','Volume']]

    else:
        # Index / commodity futures — use yfinance
        import yfinance as yf
        yf_sym = cfg["yf_symbol"]
        bars   = yf.download(yf_sym, period=f"{lookback_days+10}d",
                             auto_adjust=True, progress=False)
        if isinstance(bars.columns, pd.MultiIndex):
            bars.columns = bars.columns.get_level_values(0)
        bars = bars[['Open','High','Low','Close','Volume']].copy()
        bars.index = pd.to_datetime(bars.index).tz_localize(None)
        bars.index.name = 'Date'

    log.info(f"[{ticker}] Fetched {len(bars)} bars  "
             f"({bars.index[0].date()} → {bars.index[-1].date()})")
    return bars


def get_latest_futures_price(ticker: str) -> float:
    """Get the latest price for a futures instrument."""
    cfg = FUTURES_CONFIGS[ticker]
    try:
        if cfg["type"] == "crypto":
            client = CryptoHistoricalDataClient()
            req    = CryptoLatestQuoteRequest(symbol_or_symbols=ticker)
            quote  = client.get_crypto_latest_quote(req)
            return float(quote[ticker].ask_price)
        else:
            import yfinance as yf
            df = yf.download(cfg["yf_symbol"], period="1d",
                             interval="1m", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return float(df['Close'].iloc[-1]) if not df.empty else 0.0
    except Exception as e:
        log.warning(f"get_latest_futures_price({ticker}): {e}")
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────

def compute_futures_signal_live(ticker: str, chrom: np.ndarray) -> tuple[int, float, float]:
    """
    Compute live signal + confidence + leverage for a futures position.
    Returns (signal, confidence, leverage)
    signal: 1=LONG, -1=SHORT, 0=FLAT
    """
    cfg = FUTURES_CONFIGS[ticker]

    # Fetch fresh data via yfinance (same source as training)
    import yfinance as yf
    yf_sym = cfg["yf_symbol"]
    df = yf.download(yf_sym, period="1y", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df_ind    = add_indicators(df.copy())
    df_scaled, _ = preprocess_data(df_ind)

    decoded    = decode_futures_chromosome(chrom)
    weights    = decoded["weights"]
    thresholds = decoded["thresholds"]
    lev_gene   = decoded["leverage_genes"].mean()

    values    = df_scaled[FEATURES].values
    condition = (values > thresholds).astype(float)
    w_sum     = float(weights.sum())
    conf_arr  = (condition * weights).sum(axis=1) / max(w_sum, 1e-9)
    confidence = float(conf_arr[-1])

    signal   = 1 if confidence > 0.55 else (-1 if confidence < 0.35 else 0)
    max_lev  = float(min(decoded["max_leverage"], cfg["max_leverage"]))
    leverage = float(np.clip(1.0 + confidence * (max_lev - 1.0) * lev_gene, 1.0, max_lev))
    if confidence < 0.45:
        leverage = 1.0

    signal_names = {1: "LONG", -1: "SHORT", 0: "FLAT"}
    log.info(f"[{ticker}] Signal: {signal_names[signal]}  |  "
             f"Confidence: {confidence:.3f}  |  Leverage: {leverage:.2f}x")

    return signal, confidence, leverage


# ──────────────────────────────────────────────────────────────────────────────
# POSITION SIZING WITH LEVERAGE
# ──────────────────────────────────────────────────────────────────────────────

def calc_futures_position(confidence: float,
                           leverage: float,
                           portfolio_value: float,
                           current_price: float,
                           decoded: dict,
                           risk_guard: FuturesRiskGuard) -> int:
    """
    Calculate number of contracts/units for a futures position.
    Applies GA-evolved position scale + leverage, capped by risk guard.
    """
    allowed, reason = risk_guard.check(confidence, leverage, portfolio_value)
    if not allowed:
        log.info(f"Risk guard blocked trade: {reason}")
        return 0

    position_scale = decoded["position_scale"]
    dollar_amount  = portfolio_value * position_scale
    # Apply leverage to dollar exposure
    leveraged_amount = dollar_amount * leverage
    # Cap at risk guard maximum
    max_dollar = portfolio_value * risk_guard.MAX_POSITION_PCT
    leveraged_amount = min(leveraged_amount, max_dollar)

    units = int(leveraged_amount // current_price)
    log.info(f"Futures position: {units} units  "
             f"(scale={position_scale:.1%}, leverage={leverage:.1f}x, "
             f"exposure=${leveraged_amount:,.0f})")
    return max(units, 0)


# ──────────────────────────────────────────────────────────────────────────────
# ORDER MANAGEMENT
# ──────────────────────────────────────────────────────────────────────────────

def get_futures_position(trading_client: TradingClient, ticker: str) -> dict | None:
    """Get current futures position."""
    try:
        pos  = trading_client.get_open_position(ticker)
        qty  = int(pos.qty)
        side = "long" if qty > 0 else "short"
        return {
            'qty'          : abs(qty),
            'side'         : side,
            'entry_price'  : float(pos.avg_entry_price),
            'market_value' : float(pos.market_value),
            'unrealised_pl': float(pos.unrealized_pl),
        }
    except Exception:
        return None


def place_futures_order(trading_client: TradingClient,
                        ticker: str, qty: int,
                        side: str, reason: str = "") -> str | None:
    """Place a futures market order."""
    if qty <= 0:
        return None
    order_side = OrderSide.BUY if side in ("long", "cover") else OrderSide.SELL
    try:
        order = trading_client.submit_order(MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.GTC,  # Good Till Cancelled for futures
        ))
        action = {"long": "BUY LONG", "short": "SELL SHORT",
                  "cover": "BUY TO COVER", "exit": "SELL TO EXIT"}.get(side, side.upper())
        log.info(f"[FUTURES] {action} {qty}x {ticker}  [{reason}]  id={order.id}")
        return str(order.id)
    except Exception as e:
        log.error(f"Futures order failed ({side}): {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# INTRADAY MONITOR
# ──────────────────────────────────────────────────────────────────────────────

def monitor_futures_position(trading_client: TradingClient,
                              ticker: str,
                              entry_price: float,
                              decoded: dict,
                              side: str,
                              risk_guard: FuturesRiskGuard,
                              interval_s: int = 120) -> bool:
    """
    Monitor an open futures position with evolved stop/take-profit.
    Returns True when position is closed.
    """
    stop_mult = decoded["stop_width"]
    tp_mult   = decoded["take_profit_mult"]

    if side == "long":
        stop_price = entry_price * (1 - stop_mult)
        tp_price   = entry_price * (1 + stop_mult * tp_mult)
    else:
        stop_price = entry_price * (1 + stop_mult)
        tp_price   = entry_price * (1 - stop_mult * tp_mult)

    peak_price    = entry_price
    trailing_stop = stop_price

    log.info(f"[{ticker}] Monitoring {side.upper()}  entry={entry_price:.2f}  "
             f"stop={stop_price:.2f}  target={tp_price:.2f}")

    while True:
        time.sleep(interval_s)

        pos = get_futures_position(trading_client, ticker)
        if pos is None:
            log.info(f"[{ticker}] Position closed externally")
            return True

        current_price = get_latest_futures_price(ticker)
        if current_price == 0.0:
            continue

        # Update trailing stop
        if side == "long" and current_price > peak_price:
            peak_price    = current_price
            trailing_stop = peak_price * (1 - stop_mult * 1.5)
        elif side == "short" and current_price < peak_price:
            peak_price    = current_price
            trailing_stop = peak_price * (1 + stop_mult * 1.5)

        pnl_pct = ((current_price - entry_price) / entry_price *
                   (1 if side == "long" else -1))

        log.info(f"[{ticker}] {side.upper()} price={current_price:.2f}  "
                 f"stop={trailing_stop:.2f}  target={tp_price:.2f}  "
                 f"P&L={pnl_pct:+.2%}")

        qty = pos['qty']
        closed = False
        reason = ""

        if side == "long":
            if current_price <= trailing_stop:
                reason = "stop-loss"
                place_futures_order(trading_client, ticker, qty, "exit", reason)
                closed = True
            elif current_price >= tp_price:
                reason = "take-profit"
                place_futures_order(trading_client, ticker, qty, "exit", reason)
                closed = True
        else:
            if current_price >= trailing_stop:
                reason = "stop-loss"
                place_futures_order(trading_client, ticker, qty, "cover", reason)
                closed = True
            elif current_price <= tp_price:
                reason = "take-profit"
                place_futures_order(trading_client, ticker, qty, "cover", reason)
                closed = True

        if closed:
            risk_guard.record_trade(pnl_pct)
            alerter.trade_closed(ticker, side, qty, entry_price, current_price, reason)
            return True


# ──────────────────────────────────────────────────────────────────────────────
# MAIN DAILY SIGNAL LOOP
# ──────────────────────────────────────────────────────────────────────────────

def run_futures_signal(trading_client: TradingClient,
                       ticker: str,
                       chrom: np.ndarray,
                       decoded: dict,
                       risk_guard: FuturesRiskGuard,
                       cfg: dict) -> None:
    """Run one signal check cycle for a futures instrument."""
    log.info(f"[{ticker}] ── Signal check {datetime.now().strftime('%H:%M')} ──")

    signal, confidence, leverage = compute_futures_signal_live(ticker, chrom)
    position  = get_futures_position(trading_client, ticker)
    portfolio = float(trading_client.get_account().equity)
    price     = get_latest_futures_price(ticker)

    if price == 0.0:
        log.warning(f"[{ticker}] Could not get price — skipping")
        return

    log.info(f"[{ticker}] Portfolio=${portfolio:,.2f}  Price=${price:.2f}  "
             f"Position={position}")

    # ── LONG signal ───────────────────────────────────────────────────────────
    if signal == 1:
        if position and position['side'] == 'short':
            # Flip short → long
            log.info(f"[{ticker}] Flipping SHORT → LONG")
            place_futures_order(trading_client, ticker, position['qty'], "cover", "flip")
            time.sleep(3)
            position = None

        if position is None:
            qty = calc_futures_position(confidence, leverage, portfolio,
                                        price, decoded, risk_guard)
            if qty > 0:
                oid = place_futures_order(trading_client, ticker, qty, "long", "signal")
                if oid:
                    alerter.trade_opened(ticker, "long", qty, price, confidence)
                    time.sleep(3)
                    pos = get_futures_position(trading_client, ticker)
                    if pos:
                        monitor_futures_position(
                            trading_client, ticker,
                            pos['entry_price'], decoded, "long",
                            risk_guard, cfg.get("intraday_interval_s", 120)
                        )
        else:
            log.info(f"[{ticker}] Already LONG — monitoring")
            monitor_futures_position(
                trading_client, ticker,
                position['entry_price'], decoded, "long",
                risk_guard, cfg.get("intraday_interval_s", 120)
            )

    # ── SHORT signal ──────────────────────────────────────────────────────────
    elif signal == -1:
        if position and position['side'] == 'long':
            log.info(f"[{ticker}] Flipping LONG → SHORT")
            place_futures_order(trading_client, ticker, position['qty'], "exit", "flip")
            time.sleep(3)
            position = None

        if position is None:
            qty = calc_futures_position(confidence, leverage, portfolio,
                                        price, decoded, risk_guard)
            if qty > 0:
                oid = place_futures_order(trading_client, ticker, qty, "short", "signal")
                if oid:
                    alerter.trade_opened(ticker, "short", qty, price, confidence)
                    time.sleep(3)
                    pos = get_futures_position(trading_client, ticker)
                    if pos:
                        monitor_futures_position(
                            trading_client, ticker,
                            pos['entry_price'], decoded, "short",
                            risk_guard, cfg.get("intraday_interval_s", 120)
                        )
        else:
            log.info(f"[{ticker}] Already SHORT — monitoring")
            monitor_futures_position(
                trading_client, ticker,
                position['entry_price'], decoded, "short",
                risk_guard, cfg.get("intraday_interval_s", 120)
            )

    else:
        log.info(f"[{ticker}] FLAT signal — no trade")
        if position:
            log.info(f"[{ticker}] Holding existing {position['side']} position")


# ──────────────────────────────────────────────────────────────────────────────
# BOT LOOP
# ──────────────────────────────────────────────────────────────────────────────

def run_futures_bot(ticker: str) -> None:
    """Main futures bot loop — runs continuously."""
    if ticker not in FUTURES_CONFIGS:
        log.error(f"Unknown futures ticker: {ticker}")
        sys.exit(1)

    cfg = FUTURES_CONFIGS[ticker]

    # Setup logging to file
    fh = logging.FileHandler(cfg["log_file"], encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    log.addHandler(fh)

    log.info(f"[FUTURES BOT] Starting: {ticker} ({cfg['name']})")
    log.info(f"  Mode       : {'PAPER' if IS_PAPER else 'LIVE'}")
    log.info(f"  Max leverage: {cfg['max_leverage']}x")
    log.info(f"  Trading hours: {cfg['trading_hours']}")

    # Load chromosome
    try:
        chrom   = load_futures_chromosome(ticker)
        decoded = decode_futures_chromosome(chrom)
        log.info(f"  Evolved leverage: {decoded['max_leverage']:.1f}x")
        log.info(f"  Stop width: {decoded['stop_width']*100:.1f}%")
    except FileNotFoundError:
        log.error(f"No chromosome for {ticker}. Run: python futures_bot.py --ticker {ticker} --train")
        sys.exit(1)

    # Setup clients
    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=IS_PAPER)
    risk_guard     = FuturesRiskGuard()

    check_interval = cfg.get("intraday_interval_s", 120) * 5  # check every ~10 min

    log.info(f"[{ticker}] Bot running — checking every {check_interval//60} minutes")

    while True:
        try:
            # Reset daily risk counters at market open
            risk_guard.reset_daily()

            run_futures_signal(trading_client, ticker, chrom, decoded, risk_guard, cfg)

            log.info(f"[{ticker}] Signal complete. Sleeping {check_interval}s...")
            time.sleep(check_interval)

        except KeyboardInterrupt:
            log.info(f"[{ticker}] Futures bot stopped by user")
            break
        except Exception as e:
            log.error(f"[{ticker}] Error: {e}", exc_info=True)
            alerter.bot_crashed(ticker, str(e))
            log.info("Retrying in 5 minutes...")
            time.sleep(300)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Futures trading bot")
    parser.add_argument("--ticker", help="Futures ticker (BTC/USD, ETH/USD, ES, NQ, GC, CL)")
    parser.add_argument("--paper",  action="store_true", default=True)
    parser.add_argument("--list",   action="store_true")
    args = parser.parse_args()

    if args.list:
        print("\nTrained futures bots:")
        for ticker, cfg in FUTURES_CONFIGS.items():
            trained = Path(cfg["chromosome_file"]).exists()
            status  = "[READY]" if trained else "[needs training]"
            print(f"  {ticker:<12} {status}  {cfg['name']}")
        return

    if not args.ticker:
        parser.print_help()
        return

    run_futures_bot(args.ticker.upper())


if __name__ == "__main__":
    main()
