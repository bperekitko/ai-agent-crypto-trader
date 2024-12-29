import os
from datetime import datetime

import pandas as pd
from binance.client import Client
from .raw_data_columns import DataColumns
from .technical_indicators import add_target_to_data

DATA_PATH = os.path.dirname(__file__)
RAW_DATA_FILE_PATH = os.path.join(DATA_PATH, "raw_binance_data_BTCUSDT_1h.parquet")


def __to_data_frame(klines):
    data = []
    for kline in klines:
        date_open = datetime.fromtimestamp(kline[0] / 1000)
        open_price = float(kline[1])
        high_price = float(kline[2])
        low_price = float(kline[3])
        close_price = float(kline[4])
        volume = float(kline[5])
        date_close = datetime.fromtimestamp(kline[6] / 1000)
        quote_asset_volume = float(kline[7])
        number_of_trades = int(kline[8])
        taker_buy_base_asset_volume = float(kline[9])
        taker_buy_quote_asset_volume = float(kline[10])
        data.append(
            [
                date_open,
                open_price,
                high_price,
                low_price,
                close_price,
                volume,
                date_close,
                quote_asset_volume,
                number_of_trades,
                taker_buy_base_asset_volume,
                taker_buy_quote_asset_volume,
            ]
        )

    return pd.DataFrame(
        data,
        columns=[
            DataColumns.DATE_OPEN,
            DataColumns.OPEN,
            DataColumns.HIGH,
            DataColumns.LOW,
            DataColumns.CLOSE,
            DataColumns.VOLUME,
            DataColumns.DATE_CLOSE,
            DataColumns.QUOTE_ASSET_VOL,
            DataColumns.NUM_OF_TRADES,
            DataColumns.TAKER_BUY_BASE_ASSET_VOL,
            DataColumns.TAKER_BUY_QUOTE_ASSET_VOL,
        ],
    )


def __download_data_to_file(start_time=datetime(2021, 1, 1), end_time=datetime(2024, 5, 1)):
    client = Client()
    # Konwersja dat na milisekundy, ponieważ Binance takiego formatu używa
    start_time_ms = int(start_time.timestamp() * 1000)
    end_time_ms = int(end_time.timestamp() * 1000)

    klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, start_time_ms, end_time_ms)
    df = __to_data_frame(klines)
    df.to_parquet(RAW_DATA_FILE_PATH)


def refresh_data():
    start_time = datetime(2024, 1, 1)
    end_time = datetime(2024, 12, 31)
    print(f"Downloading BTC/USD data from {start_time} to {end_time}")
    __download_data_to_file(start_time, end_time)
    add_target_to_data(RAW_DATA_FILE_PATH)


def get_data():
    return pd.read_parquet(RAW_DATA_FILE_PATH)


def get_last_x_intervals_1h(limit: int):
    client = Client()
    historical_klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, limit=limit)
    return __to_data_frame(historical_klines)
