import os
from datetime import datetime

DATA_PATH = os.path.dirname(__file__)
RAW_DATA_FILE_PATH = os.path.join(DATA_PATH, "raw_binance_data_BTCUSDT_1h.parquet")

def path_for_data(interval: str, start:datetime, end:datetime):
    return os.path.join(DATA_PATH, f'raw_BTC_USDT_binance_{interval}_{start}-{end}')