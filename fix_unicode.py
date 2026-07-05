import re

files = ['stock_data.py', 'genetic_algorithm.py', 'train_dynamic.py']

replacements = {
    '\u2705': '[OK]',
    '\u274c': '[ERROR]',
    '\u2500': '-',
    '\u2550': '=',
    '\u2554': '+',
    '\u2557': '+',
    '\u255a': '+',
    '\u255d': '+',
    '\u2560': '+',
    '\u2563': '+',
    '\u2566': '+',
    '\u2569': '+',
    '\u256c': '+',
    '\u2014': '--',
    '\u2013': '-',
    '\u2192': '->',
    '\u00d7': 'x',
}

for fname in files:
    try:
        with open(fname, 'r', encoding='utf-8') as f:
            src = f.read()
        original = src
        for char, replacement in replacements.items():
            src = src.replace(char, replacement)
        if src != original:
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(src)
            print(f'Fixed: {fname}')
        else:
            print(f'No changes: {fname}')
    except Exception as e:
        print(f'Error on {fname}: {e}')
