import json
from pathlib import Path
f = Path('bot_configs.json')
configs = json.loads(f.read_text())
for bid, cfg in configs.items():
    if 'status' not in cfg:
        cfg['status'] = 'running'
        print(f'Fixed missing status for {bid}')
    if 'ticker' not in cfg:
        print(f'WARNING: {bid} missing ticker field')
f.write_text(json.dumps(configs, indent=2))
print('Done')
