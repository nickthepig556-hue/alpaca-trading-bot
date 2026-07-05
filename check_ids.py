import json
state = json.loads(open('bot_state.json').read())
configs = json.loads(open('bot_configs.json').read())
print('bot_state.json:')
for b in state:
    print(f"  {b['id']} -> {b['log_file']}")
print('bot_configs.json:')
for bid,b in configs.items():
    print(f"  {bid} -> {b['log_file']}")
