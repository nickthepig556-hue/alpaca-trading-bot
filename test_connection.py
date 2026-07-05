import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

load_dotenv()

API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

client = TradingClient(API_KEY, SECRET_KEY, paper=True)

account = client.get_account()
print(f"✅ Connected! Status: {account.status}")
print(f"💰 Buying Power: ${float(account.buying_power):,.2f}")
print(f"📊 Portfolio Value: ${float(account.portfolio_value):,.2f}")