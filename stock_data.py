import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# STEP 1: DEFINE YOUR STOCKS
# ============================================================
# Start with Gold ETF (GLD), easy to expand later
STOCKS = {
    'GLD':  'Gold ETF',           # Gold
    # Uncomment below to expand later:
    # 'SPY':  'S&P 500 ETF',
    # 'AAPL': 'Apple',
    # 'TSLA': 'Tesla',
    # 'BTC-USD': 'Bitcoin',
}

START_DATE = '2020-01-01'
END_DATE   = '2024-12-31'

# ============================================================
# STEP 2: DOWNLOAD HISTORICAL DATA
# ============================================================
def download_stock_data(ticker, start, end):
    print(f"Downloading {ticker} data...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):   # yfinance ≥0.2 returns MultiIndex even for 1 ticker
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    print(f"[OK] {ticker}: {len(df)} rows downloaded\n")
    return df

# ============================================================
# STEP 3: ADD TECHNICAL INDICATORS
# ============================================================
def add_indicators(df):
    # Moving Averages
    df['SMA_20']  = df['Close'].rolling(window=20).mean()   # Short term
    df['SMA_50']  = df['Close'].rolling(window=50).mean()   # Medium term
    df['SMA_200'] = df['Close'].rolling(window=min(200, len(df) - 1)).mean()

    # Relative Strength Index (RSI)
    delta     = df['Close'].diff()
    gain      = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss      = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs        = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12        = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26        = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']    = ema_12 - ema_26
    df['Signal']  = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['BB_Mid']   = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['Close'].rolling(window=20).std()

    # Daily Return
    df['Daily_Return'] = df['Close'].pct_change()

    # Volume Change
    df['Volume_Change'] = df['Volume'].pct_change()

    # Drop rows with NaN values from indicators
    df.dropna(inplace=True)

    return df

# ============================================================
# STEP 4: PREPROCESS DATA FOR GENETIC ALGORITHM
# ============================================================
def preprocess_data(df):
    # Select features for the genetic algorithm
    features = [
        'Close', 'Volume',
        'SMA_20', 'SMA_50', 'SMA_200',
        'RSI', 'MACD', 'Signal',
        'BB_Upper', 'BB_Lower',
        'Daily_Return', 'Volume_Change'
    ]
def preprocess_data(df):
    # Select features for the genetic algorithm
    features = [
        'Close', 'Volume',
        'SMA_20', 'SMA_50', 'SMA_200',
        'RSI', 'MACD', 'Signal',
        'BB_Upper', 'BB_Lower',
        'Daily_Return', 'Volume_Change'
    ]
    # Clean infinity/NaN values before scaling (common in futures data)
    import numpy as np
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Scale all features to range 0-1
    scaler    = MinMaxScaler()
    scaled    = scaler.fit_transform(df[features])
    df_scaled = pd.DataFrame(scaled, columns=features, index=df.index)

    return df_scaled, scaler

# ============================================================
# STEP 5: SAVE DATA TO CSV
# ============================================================
def save_data(df, df_scaled, ticker):
    raw_file    = f"{ticker}_raw.csv"
    scaled_file = f"{ticker}_scaled.csv"

    df.to_csv(raw_file)
    df_scaled.to_csv(scaled_file)

    print(f"[OK] Raw data saved to      : {raw_file}")
    print(f"[OK] Scaled data saved to   : {scaled_file}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    for ticker, name in STOCKS.items():
        print(f"\n{'='*50}")
        print(f"Processing: {name} ({ticker})")
        print(f"{'='*50}")

        # Download
        df = download_stock_data(ticker, START_DATE, END_DATE)

        # Add indicators
        df = add_indicators(df)

        # Preprocess
        df_scaled, scaler = preprocess_data(df)

        # Save
        save_data(df, df_scaled, ticker)

        # Preview
        print(f"\n📊 Sample of processed data:")
        print(df_scaled.tail(3).to_string())

    print(f"\n[OK] All stocks processed successfully!")