from alpaca.trading.client import TradingClient
import os
from dotenv import load_dotenv
load_dotenv()
client = TradingClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), paper=True)
account = client.get_account()
print('Equity:', account.equity)
print('Last equity:', account.last_equity)
print('PnL today:', float(account.equity) - float(account.last_equity))
print()
positions = client.get_all_positions()
for p in positions:
    print(p.symbol, p.qty, 'shares  PnL:', p.unrealized_pl)
