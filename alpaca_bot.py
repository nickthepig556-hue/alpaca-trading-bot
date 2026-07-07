"""
Step 4 – Alpaca Trading Bot
============================
• Paper trading by default — flip TRADING_MODE to 'live' when ready
• Position sizing driven by GA chromosome weight sum (confidence proxy)
• Daily GA signal at market open + intraday stop-loss / take-profit loop
• Reads GLD_best_chromosome.csv produced by genetic_algorithm.py

Setup
-----
pip install alpaca-py yfinance pandas numpy scikit-learn

Add to a .env file (never commit this):
    ALPACA_API_KEY=your_key
    ALPACA_SECRET_KEY=your_secret
    TRADING_MODE=paper          # change to 'live' when ready
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.preprocessing import MinMaxScaler
from alerts import Alerter, schedule_daily_summary
alerter = Alerter()

# ── Alpaca SDK ────────────────────────────────────────────────────────────────
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ── Local modules ─────────────────────────────────────────────────────────────
# These files must exist in the same directory:
#   stock_data.py          → add_indicators(), preprocess_data()
#   genetic_algorithm.py   → decode_chromosome(), generate_signals(), FEATURES
from stock_data import add_indicators, preprocess_data
from genetic_algorithm import decode_chromosome, generate_signals, FEATURES

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# Load from environment (set these in a .env file or your shell)
API_KEY    = os.getenv("ALPACA_API_KEY",    "YOUR_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "YOUR_SECRET_KEY")
TRADING_MODE = os.getenv("TRADING_MODE", "paper").lower()   # "paper" or "live"

TICKER = "GLD"
CHROMOSOME_FILE = f"{TICKER}_best_chromosome.csv"

BOT_CONFIG = {
    # Position sizing
    'min_allocation_pct'  : 0.05,   # 5 %  minimum position when GA confidence is low
    'max_allocation_pct'  : 0.40,   # 40 % maximum position (risk cap)
    'weight_threshold'    : 0.30,   # GA weight-sum below this → skip trade

    # Risk management
    'stop_loss_pct'       : 0.02,   # 2 % stop-loss below entry
    'take_profit_pct'     : 0.04,   # 4 % take-profit above entry
    'trailing_stop_pct'   : 0.015,  # 1.5 % trailing stop (activated after entry)

    # Timing
    'market_open_delay_s' : 300,    # wait 5 min after open before first order (avoid open volatility)
    'intraday_interval_s' : 60,     # check positions every 60 seconds
    'lookback_days'       : 60,     # days of history to fetch for signal calculation

    # Mode switch safety
    'paper_mode_check'    : True,   # refuse to go live without explicit env var
}

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    handlers=[
        logging.StreamHandler(
        stream=open(sys.stdout.fileno(), mode='w',
                    encoding='utf-8', closefd=False)
    ),
        logging.FileHandler(f"{TICKER}_bot.log", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CLIENT INITIALISATION
# ──────────────────────────────────────────────────────────────────────────────

def build_clients() -> tuple[TradingClient, StockHistoricalDataClient]:
    """Create Alpaca trading + data clients based on TRADING_MODE."""
    is_paper = (TRADING_MODE == "paper")

    if not is_paper and BOT_CONFIG['paper_mode_check']:
        confirm = os.getenv("CONFIRM_LIVE_TRADING", "").lower()
        if confirm != "yes":
            raise RuntimeError(
                "LIVE trading requested but CONFIRM_LIVE_TRADING env var is not 'yes'. "
                "Set CONFIRM_LIVE_TRADING=yes to proceed."
            )

    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=is_paper)
    data_client    = StockHistoricalDataClient(API_KEY, SECRET_KEY)

    mode_label = "[PAPER] PAPER" if is_paper else "[LIVE] LIVE"
    log.info(f"Alpaca clients initialised  [{mode_label} MODE]")
    return trading_client, data_client


# ──────────────────────────────────────────────────────────────────────────────
# CHROMOSOME LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_chromosome(path: str) -> np.ndarray:
    """Load the best GA chromosome from CSV and rebuild the flat array."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run genetic_algorithm.py first."
        )
    df     = pd.read_csv(path)
    chrom  = np.concatenate([df['weight'].values, df['threshold'].values])
    log.info(f"Chromosome loaded from {path}  ({len(chrom)} genes)")
    return chrom


# ──────────────────────────────────────────────────────────────────────────────
# MARKET DATA
# ──────────────────────────────────────────────────────────────────────────────

def fetch_recent_bars(data_client: StockHistoricalDataClient,
                      ticker: str,
                      lookback_days: int) -> pd.DataFrame:
    """Fetch recent daily bars — crypto via Alpaca, stocks via yfinance."""
    end   = pd.Timestamp.now(tz='America/New_York')
    start = end - pd.Timedelta(days=lookback_days + 10)

    CRYPTO_TICKERS = {"BTC/USD", "ETH/USD", "LTC/USD", "BCH/USD"}

    if ticker in CRYPTO_TICKERS:
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest
        crypto_client = CryptoHistoricalDataClient()
        request = CryptoBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start.isoformat(),
            end=end.isoformat(),
        )
        bars = crypto_client.get_crypto_bars(request).df
        if bars.empty:
            raise ValueError(f"No bar data returned for {ticker}")
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(ticker, level='symbol')
        bars.index = pd.to_datetime(bars.index).tz_localize(None)
        bars.index.name = 'Date'
        bars = bars.rename(columns={
            'open': 'Open', 'high': 'High',
            'low': 'Low', 'close': 'Close', 'volume': 'Volume',
        })[['Open', 'High', 'Low', 'Close', 'Volume']]

    else:
        # Free Alpaca plan blocks recent SIP data — use yfinance instead
        import yfinance as yf
        yf_ticker = ticker.replace("/", "-")
        bars = yf.download(yf_ticker,
                           period=f"{lookback_days + 10}d",
                           auto_adjust=True,
                           progress=False)
        if isinstance(bars.columns, pd.MultiIndex):
            bars.columns = bars.columns.get_level_values(0)
        if bars.empty:
            raise ValueError(f"No data returned from yfinance for {ticker}")
        bars = bars[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        bars.index = pd.to_datetime(bars.index).tz_localize(None)
        bars.index.name = 'Date'

    log.info(f"Fetched {len(bars)} bars for {ticker}  "
             f"({bars.index[0].date()} → {bars.index[-1].date()})")
    return bars

def compute_signal(bars: pd.DataFrame, chrom: np.ndarray) -> tuple[int, float]:
    """Use yfinance data for signal to match training data scaling."""
    import yfinance as yf
    ticker_yf = TICKER.replace("/", "-")   # BTC/USD → BTC-USD for yfinance
    
    try:
        df_yf = yf.download(ticker_yf, period="1y", auto_adjust=True, progress=False)
        if isinstance(df_yf.columns, pd.MultiIndex):
            df_yf.columns = df_yf.columns.get_level_values(0)
        df_ind = add_indicators(df_yf.copy())
    except Exception:
        df_ind = add_indicators(bars.copy())   # fallback to Alpaca bars

    df_scaled, _ = preprocess_data(df_ind)

    signals     = generate_signals(df_scaled, chrom)
    last_signal = int(signals.iloc[-1])

    weights, thresholds = decode_chromosome(chrom)
    last_row    = df_scaled[FEATURES].iloc[-1].values
    condition   = (last_row > thresholds).astype(float)
    confidence  = float((condition * weights).sum() / (weights.sum() + 1e-9))

    log.info(f"Signal: {'BUY' if last_signal == 1 else 'SELL/HOLD'}  |  "
             f"Confidence: {confidence:.3f}")
    return last_signal, confidence


# ──────────────────────────────────────────────────────────────────────────────
# POSITION SIZING  (GA-driven)
# ──────────────────────────────────────────────────────────────────────────────

def calc_position_size(confidence: float,
                       portfolio_value: float,
                       current_price: float,
                       config: dict) -> int:
    """
    Map GA confidence → allocation percentage → number of shares.

        allocation_pct = min_pct + confidence * (max_pct - min_pct)

    Higher GA confidence (more features above thresholds, heavier weights)
    → larger position size.
    """
    if confidence < config['weight_threshold']:
        log.info(f"Confidence {confidence:.3f} below threshold "
                 f"{config['weight_threshold']} — skipping trade")
        return 0

    min_p = config['min_allocation_pct']
    max_p = config['max_allocation_pct']
    alloc_pct  = min_p + confidence * (max_p - min_p)
    alloc_pct  = min(alloc_pct, max_p)

    dollar_amount = portfolio_value * alloc_pct
    shares        = int(dollar_amount // current_price)

    log.info(f"Position size: {shares} shares  "
             f"(confidence={confidence:.3f}, alloc={alloc_pct:.1%}, "
             f"${dollar_amount:,.0f} of ${portfolio_value:,.0f})")
    return shares


# ──────────────────────────────────────────────────────────────────────────────
# ORDER MANAGEMENT
# ──────────────────────────────────────────────────────────────────────────────

def get_position(trading_client: TradingClient, ticker: str) -> dict | None:
    """Return current position dict including side (long/short), or None if flat."""
    try:
        pos = trading_client.get_open_position(ticker)
        qty = int(pos.qty)
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


def get_portfolio_value(trading_client: TradingClient) -> float:
    """Return total portfolio equity."""
    account = trading_client.get_account()
    return float(account.equity)


def place_buy(trading_client: TradingClient, ticker: str, qty: int,
              side: str = "long") -> str | None:
    """Submit a market order — long (buy) or short (sell short)."""
    if qty <= 0:
        return None
    order_side = OrderSide.BUY if side == "long" else OrderSide.SELL
    try:
        order = trading_client.submit_order(
            MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
        )
        action = "BUY (long)" if side == "long" else "SELL SHORT"
        log.info(f"[OK] {action} order submitted: {qty} x {ticker}  [id={order.id}]")
        return str(order.id)
    except Exception as e:
        log.error(f"Order failed ({side}): {e}")
        return None


def place_sell(trading_client: TradingClient, ticker: str, qty: int,
               reason: str = "signal", side: str = "long") -> str | None:
    """Close a position — sell long or buy to cover short."""
    if qty <= 0:
        return None
    order_side = OrderSide.SELL if side == "long" else OrderSide.BUY
    try:
        order = trading_client.submit_order(
            MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
        )
        action = "SELL" if side == "long" else "BUY TO COVER"
        log.info(f"[CLOSE] {action} ({reason}): {qty} x {ticker}  [id={order.id}]")
        return str(order.id)
    except Exception as e:
        log.error(f"Close order failed ({side}): {e}")
        return None

def cancel_open_orders(trading_client: TradingClient, ticker: str) -> None:
    """Cancel any pending orders for the ticker before placing a new one."""
    try:
        orders = trading_client.get_orders()
        for o in orders:
            if o.symbol == ticker and o.status in (
                OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING_NEW
            ):
                trading_client.cancel_order_by_id(o.id)
                log.info(f"Cancelled pending order {o.id}")
    except Exception as e:
        log.warning(f"Could not cancel orders: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# INTRADAY STOP-LOSS / TAKE-PROFIT MONITOR
# ──────────────────────────────────────────────────────────────────────────────

def get_latest_price(data_client, ticker: str) -> float:
    """Fetch latest price — crypto via Alpaca, stocks via yfinance."""
    CRYPTO_TICKERS = {"BTC/USD", "ETH/USD", "LTC/USD", "BCH/USD"}
    try:
        if ticker in CRYPTO_TICKERS:
            from alpaca.data.historical import CryptoHistoricalDataClient
            from alpaca.data.requests import CryptoLatestQuoteRequest
            client = CryptoHistoricalDataClient()
            req    = CryptoLatestQuoteRequest(symbol_or_symbols=ticker)
            quote  = client.get_crypto_latest_quote(req)
            return float(quote[ticker].ask_price)
        else:
            import yfinance as yf
            yf_ticker = ticker.replace("/", "-")
            df = yf.download(yf_ticker, period="1d",
                             interval="1m", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty:
                return 0.0
            return float(df['Close'].iloc[-1])
    except Exception as e:
        log.warning(f"get_latest_price failed: {e}")
        return 0.0

def monitor_position(trading_client, data_client, ticker, entry_price, config, side="long"):
    """
    Intraday loop: check current price against stop-loss and take-profit.
    Returns True if position was closed, False if still open.
    """
    stop_price   = entry_price * (1 - config['stop_loss_pct'])
    target_price = entry_price * (1 + config['take_profit_pct'])
    trailing_stop = None   # disabled until trade moves in our favour
    peak_price    = entry_price
    trailing_activated = False

    log.info(f"Monitoring {ticker}  |  entry={entry_price:.2f}  "
             f"stop={stop_price:.2f}  target={target_price:.2f}")

    while True:
        time.sleep(config['intraday_interval_s'])

        pos = get_position(trading_client, ticker)
        if pos is None:
            log.info("Position already closed (possibly by another order)")
            return True

        current_price = get_latest_price(data_client, ticker)
        if current_price == 0.0:
            continue

        side = pos['side']

        # Recalculate stop and target based on side
        if side == 'short':
            stop_price   = entry_price * (1 + config['stop_loss_pct'])
            target_price = entry_price * (1 - config['take_profit_pct'])
            # Only activate trailing stop after price moves 1x stop distance in our favour
            if current_price < entry_price * (1 - config['stop_loss_pct']):
                trailing_activated = True
            if trailing_activated and current_price < peak_price:
                peak_price    = current_price
                trailing_stop = peak_price * (1 + config['trailing_stop_pct'])
        else:
            stop_price   = entry_price * (1 - config['stop_loss_pct'])
            target_price = entry_price * (1 + config['take_profit_pct'])
            # Only activate trailing stop after price moves 1x stop distance in our favour
            if current_price > entry_price * (1 + config['stop_loss_pct']):
                trailing_activated = True
            if trailing_activated and current_price > peak_price:
                peak_price    = current_price
                trailing_stop = peak_price * (1 - config['trailing_stop_pct'])

        log.info(f"  {ticker} [{side}] price={current_price:.2f}  "
                 f"stop={stop_price:.2f}  trailing={trailing_stop:.2f if trailing_stop else 'inactive'}  "
                 f"target={target_price:.2f}  P&L={pos['unrealised_pl']:+.2f}")

        qty = pos['qty']

        if side == 'short':
            if current_price >= stop_price:
                log.warning(f"[STOP] SHORT STOP-LOSS at {current_price:.2f}")
                place_sell(trading_client, ticker, qty,
                           reason="stop-loss", side="short")
                alerter.trade_closed(ticker, "short", qty, entry_price, current_price, "stop-loss")
                return True
            if trailing_activated and trailing_stop and current_price >= trailing_stop:
                log.warning(f"[STOP] SHORT TRAILING STOP at {current_price:.2f}")
                place_sell(trading_client, ticker, qty,
                           reason="trailing-stop", side="short")
                alerter.trade_closed(ticker, "short", qty, entry_price, current_price, "trailing-stop")
                return True
            if current_price <= target_price:
                log.info(f"[TARGET] SHORT TAKE-PROFIT at {current_price:.2f}")
                place_sell(trading_client, ticker, qty,
                           reason="take-profit", side="short")
                alerter.trade_closed(ticker, "short", qty, entry_price, current_price, "take-profit")
                return True
        else:
            if current_price <= stop_price:
                log.warning(f"[STOP] STOP-LOSS at {current_price:.2f}")
                place_sell(trading_client, ticker, qty,
                           reason="stop-loss", side="long")
                alerter.trade_closed(ticker, "long", qty, entry_price, current_price, "stop-loss")
                return True
            if trailing_activated and trailing_stop and current_price <= trailing_stop:
                log.warning(f"[STOP] TRAILING STOP at {current_price:.2f}")
                place_sell(trading_client, ticker, qty,
                           reason="trailing-stop", side="long")
                alerter.trade_closed(ticker, "long", qty, entry_price, current_price, "trailing-stop")
                return True
            if current_price >= target_price:
                log.info(f"[TARGET] TAKE-PROFIT at {current_price:.2f}")
                place_sell(trading_client, ticker, qty,
                           reason="take-profit", side="long")
                alerter.trade_closed(ticker, "long", qty, entry_price, current_price, "take-profit")
                return True


# ──────────────────────────────────────────────────────────────────────────────
# MARKET HOURS HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def is_market_open(trading_client: TradingClient) -> bool:
    clock = trading_client.get_clock()
    return clock.is_open


def wait_for_market_open(trading_client: TradingClient) -> None:
    """Block until market is open — skips wait for crypto (trades 24/7)."""
    CRYPTO_TICKERS = {"BTC/USD", "ETH/USD", "LTC/USD", "BCH/USD"}
    if TICKER in CRYPTO_TICKERS:
        log.info(f"{TICKER} is crypto — trades 24/7, skipping market hours check")
        return
    while True:
        clock = trading_client.get_clock()
        if clock.is_open:
            return
        wait_s = (clock.next_open - clock.timestamp).total_seconds() + 5
        wait_s = max(60, min(wait_s, 3600))
        log.info(f"Market closed. Next open in {wait_s/60:.0f} min. Sleeping...")
        time.sleep(wait_s)

# ──────────────────────────────────────────────────────────────────────────────
# DAILY SIGNAL EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

def run_daily_signal(trading_client: TradingClient,
                     data_client: StockHistoricalDataClient,
                     chrom: np.ndarray,
                     config: dict) -> None:
    """
    Core daily routine — handles long AND short positions.
      Signal BUY  → open long  (or flip short → long)
      Signal SELL → open short (or flip long → short)
    """
    log.info("=" * 55)
    log.info(f"Daily signal run  [{datetime.now().strftime('%Y-%m-%d %H:%M')}]")
    log.info("=" * 55)

    cancel_open_orders(trading_client, TICKER)

    # ── Compute GA signal ──────────────────────────────────────────────────
    bars = fetch_recent_bars(data_client, TICKER, config['lookback_days'])
    signal, confidence = compute_signal(bars, chrom)

    # ── Current state ──────────────────────────────────────────────────────
    position        = get_position(trading_client, TICKER)
    portfolio_value = get_portfolio_value(trading_client)
    current_price   = float(bars['Close'].iloc[-1])

    log.info(f"Portfolio value : ${portfolio_value:,.2f}")
    log.info(f"Current price   : ${current_price:.2f}")
    log.info(f"Open position   : {position}")

    # ── BUY signal ────────────────────────────────────────────────────────
    if signal == 1:
        if position and position['side'] == 'short':
            # Flip: close short first
            log.info("Flipping short → long: closing short position")
            place_sell(trading_client, TICKER, position['qty'],
                      reason="flip to long", side="short")
            time.sleep(5)
            position = None

        if position is None:
            qty = calc_position_size(confidence, portfolio_value,
                                     current_price, config)
            if qty > 0:
                order_id = place_buy(trading_client, TICKER, qty, side="long")
                if order_id:
                    alerter.trade_opened(TICKER, "long", qty, current_price, confidence)
                    time.sleep(5)
                    position = get_position(trading_client, TICKER)
                    if position:
                        monitor_position(trading_client, data_client,
                                        TICKER, position['entry_price'],
                                        config, position['side'])
        else:
            log.info("BUY signal — already long, monitoring existing position")
            monitor_position(trading_client, data_client, TICKER,
                            position['entry_price'], config, position['side'])

    # ── SELL signal ───────────────────────────────────────────────────────
    elif signal == -1:
        if position and position['side'] == 'long':
            # Flip: close long first
            log.info("Flipping long → short: closing long position")
            place_sell(trading_client, TICKER, position['qty'],
                      reason="flip to short", side="long")
            time.sleep(5)
            position = None

        if position is None:
            # Open short
            qty = calc_position_size(confidence, portfolio_value,
                                     current_price, config)
            if qty > 0:
                log.info(f"Opening SHORT position: {qty} x {TICKER}")
                order_id = place_buy(trading_client, TICKER, qty, side="short")
                if order_id:
                    alerter.trade_opened(TICKER, "short", qty, current_price, confidence)
                    time.sleep(5)
                    position = get_position(trading_client, TICKER)
                    if position:
                        monitor_position(trading_client, data_client,
                                        TICKER, position['entry_price'],
                                        config, position['side'])
        else:
            log.info("SELL signal — already short, monitoring existing position")
            monitor_position(trading_client, data_client, TICKER,
                            position['entry_price'], config, position['side'])


# ──────────────────────────────────────────────────────────────────────────────
# PAPER → LIVE SWITCH GUIDE  (printed on startup in paper mode)
# ──────────────────────────────────────────────────────────────────────────────

SWITCH_GUIDE = """
┌─────────────────────────────────────────────────────┐
│  HOW TO SWITCH FROM PAPER → LIVE TRADING            │
├─────────────────────────────────────────────────────┤
│  1. Log in to alpaca.markets and fund your account  │
│  2. Generate LIVE API keys (separate from paper)    │
│  3. Update your .env file:                          │
│       ALPACA_API_KEY=<live_key>                     │
│       ALPACA_SECRET_KEY=<live_secret>               │
│       TRADING_MODE=live                             │
│       CONFIRM_LIVE_TRADING=yes                      │
│  4. Back-test thoroughly before going live          │
│  5. Start with small position sizes (lower          │
│       max_allocation_pct in BOT_CONFIG)             │
└─────────────────────────────────────────────────────┘
"""


def run_bot() -> None:
    """
    Outer loop: runs every trading day.
    """
    log.info("[BOT] Alpaca GA Trading Bot starting up")
    log.info(f"   Mode    : {TRADING_MODE.upper()}")
    log.info(f"   Ticker  : {TICKER}")
    log.info(f"   Chromosome : {CHROMOSOME_FILE}")

    trading_client, data_client = build_clients()
    chrom = load_chromosome(CHROMOSOME_FILE)

    while True:
        try:
            wait_for_market_open(trading_client)

            delay = BOT_CONFIG['market_open_delay_s']
            log.info(f"Market open — waiting {delay}s before first signal...")
            time.sleep(delay)

            run_daily_signal(trading_client, data_client, chrom, BOT_CONFIG)

            log.info("Signal check complete. Sleeping 5 minutes until next check.")
            time.sleep(5 * 60)

        except KeyboardInterrupt:
            log.info("Bot stopped by user (KeyboardInterrupt)")
            break
        except Exception as e:
            log.error(f"Unexpected error: {e}", exc_info=True)
            log.info("Retrying in 5 minutes...")
            time.sleep(300)


if __name__ == "__main__":
    run_bot()
    schedule_daily_summary(alerter)