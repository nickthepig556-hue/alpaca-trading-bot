"""
fix_bot.py  —  run once to patch alpaca_bot.py
Fixes:
  1. UnicodeEncodeError on Windows (emoji in log messages)
  2. Diagnoses / confirms .env API key loading
  3. Verifies Alpaca credentials against the paper API
"""

import os, re, sys

# ── 1. CHECK .env EXISTS AND KEYS ARE PRESENT ─────────────────────────────────
print("=" * 55)
print("STEP 1: Checking .env file")
print("=" * 55)

env_path = os.path.join(os.path.dirname(__file__), ".env")
if not os.path.exists(env_path):
    print(f"ERROR: .env file not found at {env_path}")
    print("Create it with:")
    print("  ALPACA_API_KEY=your_key")
    print("  ALPACA_SECRET_KEY=your_secret")
    print("  TRADING_MODE=paper")
    sys.exit(1)

with open(env_path) as f:
    env_lines = f.read()

print(f"Found .env at: {env_path}")

# Parse key/value pairs
env_vars = {}
for line in env_lines.splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        env_vars[k.strip()] = v.strip()

api_key    = env_vars.get("ALPACA_API_KEY", "")
secret_key = env_vars.get("ALPACA_SECRET_KEY", "")
mode       = env_vars.get("TRADING_MODE", "paper")

print(f"  TRADING_MODE      : {mode}")
print(f"  ALPACA_API_KEY    : {api_key[:6]}{'*'*10 if len(api_key) > 6 else ' (MISSING)'}")
print(f"  ALPACA_SECRET_KEY : {secret_key[:4]}{'*'*10 if len(secret_key) > 4 else ' (MISSING)'}")

if not api_key or api_key in ("your_key", "YOUR_API_KEY", "<live_key>"):
    print("\nERROR: ALPACA_API_KEY looks like a placeholder or is missing.")
    print("Go to https://app.alpaca.markets/paper-trading/overview")
    print("Click 'API Keys' -> 'Generate New Key' and paste the real values into .env")
    sys.exit(1)

if not secret_key or secret_key in ("your_secret", "YOUR_SECRET_KEY", "<live_secret>"):
    print("\nERROR: ALPACA_SECRET_KEY looks like a placeholder or is missing.")
    sys.exit(1)

# ── 2. TEST THE CREDENTIALS DIRECTLY ─────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2: Testing credentials against Alpaca paper API")
print("=" * 55)

try:
    import requests
    resp = requests.get(
        "https://paper-api.alpaca.markets/v2/account",
        headers={
            "APCA-API-KEY-ID"    : api_key,
            "APCA-API-SECRET-KEY": secret_key,
        },
        timeout=10,
    )
    if resp.status_code == 200:
        data = resp.json()
        print(f"  Credentials VALID")
        print(f"  Account status : {data.get('status')}")
        print(f"  Portfolio value: ${float(data.get('portfolio_value', 0)):,.2f}")
        print(f"  Buying power   : ${float(data.get('buying_power', 0)):,.2f}")
    elif resp.status_code == 401:
        print("  ERROR 401 — credentials are wrong.")
        print("  Either the key/secret is incorrect, or you're using")
        print("  LIVE keys with TRADING_MODE=paper (they are separate).")
        print("\n  Fix: go to https://app.alpaca.markets/paper-trading/overview")
        print("  and regenerate PAPER API keys, then update .env")
        sys.exit(1)
    else:
        print(f"  Unexpected response: {resp.status_code} — {resp.text[:200]}")
        sys.exit(1)
except requests.exceptions.ConnectionError:
    print("  ERROR: No internet connection or Alpaca is unreachable.")
    sys.exit(1)

# ── 3. PATCH alpaca_bot.py ────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3: Patching alpaca_bot.py")
print("=" * 55)

bot_path = os.path.join(os.path.dirname(__file__), "alpaca_bot.py")
if not os.path.exists(bot_path):
    print(f"ERROR: alpaca_bot.py not found at {bot_path}")
    sys.exit(1)

with open(bot_path, encoding="utf-8") as f:
    source = f.read()

original = source

# ── Patch A: add load_dotenv at top if missing ────────────────────────────────
if "load_dotenv" not in source:
    source = source.replace(
        "import os",
        "import os\nfrom dotenv import load_dotenv\nload_dotenv()",
        1,
    )
    print("  + Added load_dotenv() at top")
else:
    print("  . load_dotenv already present")

# ── Patch B: fix logging StreamHandler encoding ───────────────────────────────
old_handler = "logging.StreamHandler()"
new_handler = (
    "logging.StreamHandler(\n"
    "        stream=open(sys.stdout.fileno(), mode='w',\n"
    "                    encoding='utf-8', closefd=False)\n"
    "    )"
)

if old_handler in source:
    # Make sure sys is imported
    if "import sys" not in source:
        source = source.replace("import os", "import os\nimport sys", 1)
    source = source.replace(old_handler, new_handler, 1)
    print("  + Fixed StreamHandler encoding (UTF-8)")
else:
    print("  . StreamHandler already patched or not found")

# ── Patch C: add encoding to FileHandler ─────────────────────────────────────
old_fh = f'logging.FileHandler(f"{"{TICKER}"}_bot.log")'
new_fh = f'logging.FileHandler(f"{"{TICKER}"}_bot.log", encoding="utf-8")'

# More robust: regex replace any FileHandler without encoding kwarg
fh_pattern = r'logging\.FileHandler\(([^,)]+)\)(?!.*encoding)'
fh_replacement = r'logging.FileHandler(\1, encoding="utf-8")'
patched, n = re.subn(fh_pattern, fh_replacement, source)
if n:
    source = patched
    print("  + Added encoding='utf-8' to FileHandler")
else:
    print("  . FileHandler already has encoding or not found")

# ── Patch D: replace emoji characters with ASCII equivalents ─────────────────
replacements = {
    "🤖": "[BOT]",
    "📄": "[PAPER]",
    "🔴": "[LIVE]",
    "✅": "[OK]",
    "⛔": "[STOP]",
    "🎯": "[TARGET]",
    "📊": "[DATA]",
}
n_replaced = 0
for emoji, text in replacements.items():
    if emoji in source:
        source = source.replace(emoji, text)
        n_replaced += 1
if n_replaced:
    print(f"  + Replaced {n_replaced} emoji(s) with ASCII equivalents")
else:
    print("  . No emoji found (already clean)")

# ── Write patched file ────────────────────────────────────────────────────────
if source != original:
    backup_path = bot_path.replace(".py", "_backup.py")
    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(original)
    with open(bot_path, "w", encoding="utf-8") as f:
        f.write(source)
    print(f"\n  Backup saved to : {backup_path}")
    print(f"  Patched file    : {bot_path}")
else:
    print("\n  No changes needed — file already up to date")

print("\n" + "=" * 55)
print("All checks passed. Run: python alpaca_bot.py")
print("=" * 55)
