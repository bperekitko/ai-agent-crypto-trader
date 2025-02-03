import os
from enum import Enum

from dotenv import load_dotenv


class Environment(Enum):
    TEST = 0
    PROD = 1


load_dotenv()
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../logs')


class Config:
    def __init__(self, env: Environment):
        self.env = env
        self.binance_futures_api_key = os.getenv(f'{self.env.name}_BINANCE_FUTURES_API_KEY')
        self.binance_futures_secret_key = os.getenv(f'{self.env.name}_BINANCE_FUTURES_SECRET_KEY')
