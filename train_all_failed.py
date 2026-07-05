import json, subprocess, sys
from pathlib import Path
configs = json.loads(Path('bot_configs.json').read_text())
for bot_id, cfg in configs.items():
    if cfg['status'] == 'training_failed':
        tmp = Path(f'_train_{bot_id}.json')
        tmp.write_text(json.dumps(cfg))
        print(f'Training {cfg["ticker"]}...')
        result = subprocess.run([sys.executable, 'train_dynamic.py', '--config', str(tmp)], text=True)
        tmp.unlink(missing_ok=True)
        cfg['status'] = 'running' if result.returncode == 0 else 'training_failed'
        configs[bot_id] = cfg
        print(f'{cfg["ticker"]}: OK' if result.returncode == 0 else f'{cfg["ticker"]}: FAILED')
Path('bot_configs.json').write_text(json.dumps(configs, indent=2))
print('All done')
