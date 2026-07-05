import json
configs = json.loads(open('bot_configs.json').read())
for bid, cfg in configs.items():
    thresh = cfg.get('risk', {}).get('weight_threshold', 0.30)
    print(f"{cfg['ticker']:<12} threshold={thresh}")
