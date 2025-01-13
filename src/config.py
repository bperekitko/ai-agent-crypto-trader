import os
from dotenv import load_dotenv

load_dotenv()

BINANCE_FUTURES_API_KEY = os.getenv("BINANCE_FUTURES_API_KEY")
BINANCE_FUTURES_SECRET_KEY = os.getenv("BINANCE_FUTURES_SECRET_KEY")

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
