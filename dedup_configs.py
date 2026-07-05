import json
from pathlib import Path

configs = json.loads(Path('bot_configs.json').read_text())
seen = {}
for bid, cfg in configs.items():
    ticker = cfg['ticker']
    if ticker not in seen:
        seen[ticker] = (bid, cfg)
    else:
        print(f'Removing duplicate: {ticker} ({bid})')

deduped = {bid: cfg for ticker, (bid, cfg) in seen.items()}
Path('bot_configs.json').write_text(json.dumps(deduped, indent=2))
print(f'Done — {len(deduped)} unique bots remaining')
