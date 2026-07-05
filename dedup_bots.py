import json
from pathlib import Path

configs = json.loads(Path('bot_configs.json').read_text())

# Get all tickers already in bot_configs.json
existing_tickers = {cfg['ticker'] for cfg in configs.values()}
print(f'Tickers in bot_configs.json: {existing_tickers}')

# Load bot_state.json and remove any that are already in bot_configs.json
state = json.loads(Path('bot_state.json').read_text())
kept = [b for b in state if b['ticker'] not in existing_tickers]
removed = [b for b in state if b['ticker'] in existing_tickers]

print(f'Removing from bot_state.json: {[b["ticker"] for b in removed]}')
print(f'Keeping in bot_state.json: {[b["ticker"] for b in kept]}')

Path('bot_state.json').write_text(json.dumps(kept, indent=2))
print('Done')
