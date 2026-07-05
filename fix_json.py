import re
with open('bot_configs.json', 'r', encoding='utf-8') as f:
    content = f.read()
fixed = re.sub(r',(\s*[}\]])', r'\1', content)
with open('bot_configs.json', 'w', encoding='utf-8') as f:
    f.write(fixed)
import json
try:
    json.loads(fixed)
    print('JSON fixed and valid')
except Exception as e:
    print(f'Still broken: {e}')
