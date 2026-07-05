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
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.preprocessing import MinMaxScaler

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
        logging.StreamHandler(),
        logging.FileHandler(f"{TICKER}_bot.log"),
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

    mode_label = "📄 PAPER" if is_paper else "🔴 LIVE"
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
    """Fetch recent daily bars from Alpaca and return a raw OHLCV DataFrame."""
    end   = pd.Timestamp.now(tz='America/New_York')
    start = end - pd.Timedelta(days=lookback_days + 10)   # buffer for weekends/holidays

    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,
        start=start.isoformat(),
        end=end.isoformat(),
    )
    bars = data_client.get_stock_bars(request).df

    if bars.empty:
        raise ValueError(f"No bar data returned for {ticker}")

    # Alpaca returns MultiIndex (symbol, timestamp) — flatten to plain DatetimeIndex
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(ticker, level='symbol')

    bars.index = pd.to_datetime(bars.index).tz_localize(None)
    bars.index.name = 'Date'

    # Rename to match stock_data.py conventions
    bars = bars.rename(columns={
        'open'  : 'Open',
        'high'  : 'High',
        'low'   : 'Low',
        'close' : 'Close',
        'volume': 'Volume',
    })[['Open', 'High', 'Low', 'Close', 'Volume']]

    log.info(f"Fetched {len(bars)} bars for {ticker}  "
             f"({bars.index[0].date()} → {bars.index[-1].date()})")
    return bars


def compute_signal(bars: pd.DataFrame, chrom: np.ndarray) -> tuple[int, float]:
    """
    Run the full indicator + scaling + GA signal pipeline on fresh bar data.

    Returns
    -------
    signal      : 1 (buy) or -1 (sell/hold)
    confidence  : GA weight-sum normalised to [0, 1]  — used for position sizing
    """
    df_ind = add_indicators(bars.copy())
    df_scaled, _ = preprocess_data(df_ind)

    signals    = generate_signals(df_scaled, chrom)
    last_signal = int(signals.iloc[-1])

    # Confidence: sum of weights for features whose value exceeded their threshold
    weights, thresholds = decode_chromosome(chrom)
    last_row   = df_scaled[FEATURES].iloc[-1].values
    condition  = (last_row > thresholds).astype(float)
    confidence = float((condition * weights).sum() / (weights.sum() + 1e-9))

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
    """Return current position dict or None if flat."""
    try:
        pos = trading_client.get_open_position(ticker)
        return {
            'qty'        : int(pos.qty),
            'entry_price': float(pos.avg_entry_price),
            'market_value': float(pos.market_value),
            'unrealised_pl': float(pos.unrealized_pl),
        }
    except Exception:
        return None


def get_portfolio_value(trading_client: TradingClient) -> float:
    """Return total portfolio equity."""
    account = trading_client.get_account()
    return float(account.equity)


def place_buy(trading_client: TradingClient, ticker: str, qty: int) -> str | None:
    """Submit a market buy order. Returns order ID or None on failure."""
    if qty <= 0:
        return None
    try:
        order = trading_client.submit_order(
            MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
        )
        log.info(f"✅ BUY order submitted: {qty} x {ticker}  [id={order.id}]")
        return str(order.id)
    except Exception as e:
        log.error(f"BUY order failed: {e}")
        return None


def place_sell(trading_client: TradingClient, ticker: str, qty: int,
               reason: str = "signal") -> str | None:
    """Submit a market sell order."""
    if qty <= 0:
        return None
    try:
        order = trading_client.submit_order(
            MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
        )
        log.info(f"🔴 SELL order submitted ({reason}): {qty} x {ticker}  [id={order.id}]")
        return str(order.id)
    except Exception as e:
        log.error(f"SELL order failed: {e}")
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

def get_latest_price(data_client: StockHistoricalDataClient, ticker: str) -> float:
    """Fetch the most recent bar's close price for intraday monitoring."""
    end   = pd.Timestamp.now(tz='America/New_York')
    start = end - pd.Timedelta(minutes=10)
    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Minute,
        start=start.isoformat(),
        end=end.isoformat(),
    )
    bars = data_client.get_stock_bars(request).df
    if bars.empty:
        return 0.0
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(ticker, level='symbol')
    return float(bars['close'].iloc[-1])


def monitor_position(trading_client: TradingClient,
                     data_client: StockHistoricalDataClient,
                     ticker: str,
                     entry_price: float,
                     config: dict) -> bool:
    """
    Intraday loop: check current price against stop-loss and take-profit.
    Returns True if position was closed, False if still open.
    """
    stop_price   = entry_price * (1 - config['stop_loss_pct'])
    target_price = entry_price * (1 + config['take_profit_pct'])
    trailing_stop = entry_price   # will be updated as price rises
    peak_price    = entry_price

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

        # Update trailing stop
        if current_price > peak_price:
            peak_price    = current_price
            trailing_stop = peak_price * (1 - config['trailing_stop_pct'])

        log.info(f"  {ticker} price={current_price:.2f}  "
                 f"trailing_stop={trailing_stop:.2f}  target={target_price:.2f}  "
                 f"P&L={pos['unrealised_pl']:+.2f}")

        qty = pos['qty']

        if current_price <= stop_price:
            log.warning(f"⛔ STOP-LOSS triggered at {current_price:.2f}")
            place_sell(trading_client, ticker, qty, reason="stop-loss")
            return True

        if current_price <= trailing_stop and current_price < peak_price:
            log.warning(f"⛔ TRAILING STOP triggered at {current_price:.2f}")
            place_sell(trading_client, ticker, qty, reason="trailing-stop")
            return True

        if current_price >= target_price:
            log.info(f"🎯 TAKE-PROFIT triggered at {current_price:.2f}")
            place_sell(trading_client, ticker, qty, reason="take-profit")
            return True


# ──────────────────────────────────────────────────────────────────────────────
# MARKET HOURS HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def is_market_open(trading_client: TradingClient) -> bool:
    clock = trading_client.get_clock()
    return clock.is_open


def wait_for_market_open(trading_client: TradingClient) -> None:
    """Block until Alpaca reports the market is open."""
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
    Core daily routine:
      1. Fetch recent bars → compute GA signal
      2. If BUY and no position → size + enter
      3. If SELL and in position → exit
      4. Start intraday stop/target monitor until market close
    """
    log.info("=" * 55)
    log.info(f"Daily signal run  [{datetime.now().strftime('%Y-%m-%d %H:%M')}]")
    log.info("=" * 55)

    cancel_open_orders(trading_client, TICKER)

    # ── Compute GA signal ─────────────────────────────────────────────────
    bars = fetch_recent_bars(data_client, TICKER, config['lookback_days'])
    signal, confidence = compute_signal(bars, chrom)

    # ── Current state ─────────────────────────────────────────────────────
    position = get_position(trading_client, TICKER)
    portfolio_value = get_portfolio_value(trading_client)
    current_price = float(bars['Close'].iloc[-1])

    log.info(f"Portfolio value : ${portfolio_value:,.2f}")
    log.info(f"Current price   : ${current_price:.2f}")
    log.info(f"Open position   : {position}")

    # ── Act on signal ─────────────────────────────────────────────────────
    if signal == 1 and position is None:
        # BUY signal, no current position
        qty = calc_position_size(confidence, portfolio_value, current_price, config)
        if qty > 0:
            order_id = place_buy(trading_client, TICKER, qty)
            if order_id:
                time.sleep(5)   # let order fill
                position = get_position(trading_client, TICKER)
                if position:
                    entry = position['entry_price']
                    monitor_position(trading_client, data_client, TICKER, entry, config)

    elif signal == -1 and position is not None:
        # SELL signal, close existing position
        place_sell(trading_client, TICKER, position['qty'], reason="GA signal")

    else:
        if signal == 1 and position is not None:
            log.info("BUY signal but already in position — monitoring existing trade")
            monitor_position(trading_client, data_client, TICKER,
                             position['entry_price'], config)
        else:
            log.info("SELL signal and no position — nothing to do")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN BOT LOOP
# ──────────────────────────────────────────────────────────────────────────────

def run_bot() -> None:
    """
    Outer loop: runs every trading day.
      • Waits for market open
      • Pauses market_open_delay_s seconds (avoids open auction volatility)
      • Runs daily signal logic
      • Sleeps until next trading day
    """
    log.info("🤖 Alpaca GA Trading Bot starting up")
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

            # Sleep until next market open check (~22 hours)
            log.info("Daily run complete. Sleeping 22 hours until next check.")
            time.sleep(22 * 3600)

        except KeyboardInterrupt:
            log.info("Bot stopped by user (KeyboardInterrupt)")
            break
        except Exception as e:
            log.error(f"Unexpected error: {e}", exc_info=True)
            log.info("Retrying in 5 minutes...")
            time.sleep(300)


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


if __name__ == "__main__":
    if TRADING_MODE == "paper":
        print(SWITCH_GUIDE)
    run_bot()
