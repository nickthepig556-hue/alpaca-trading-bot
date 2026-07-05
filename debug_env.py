import os
from dotenv import load_dotenv

load_dotenv()

api_key    = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")
base_url   = os.getenv("ALPACA_BASE_URL")

print(f"API Key    : {api_key}")
print(f"Secret Key : {secret_key}")
print(f"Base URL   : {base_url}")