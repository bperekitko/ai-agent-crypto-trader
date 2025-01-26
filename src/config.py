import os

from dotenv import load_dotenv

load_dotenv()
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../logs')

ENVIRONMENT = 'TEST' if os.getenv("ENVIRONMENT") is None else os.getenv('ENVIRONMENT')
IS_TEST = ENVIRONMENT != 'PROD'

BINANCE_FUTURES_API_KEY = os.getenv(f'{ENVIRONMENT}_BINANCE_FUTURES_API_KEY')
BINANCE_FUTURES_SECRET_KEY = os.getenv(f'{ENVIRONMENT}_BINANCE_FUTURES_SECRET_KEY')

BINANCE_FUTURES_API_URL = 'https://testnet.binancefuture.com' if IS_TEST else 'https://fapi.binance.com'
BINANCE_FUTURES_WEBSOCKET_API_URL = 'wss://stream.binancefuture.com' if IS_TEST else 'wss://fstream.binance.com'
