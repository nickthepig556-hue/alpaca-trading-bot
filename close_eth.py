from alpaca.trading.client import TradingClient
import os
from dotenv import load_dotenv
load_dotenv()
client = TradingClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), paper=True)
try:
    client.close_position('ETHUSD')
    print('ETH position closed')
except Exception as e:
    print('Error:', e)
