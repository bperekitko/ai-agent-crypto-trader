import os

import pandas as pd

from data.exchange.binance.binance_client import BinanceClient
from model.evaluation.time_series_cross_validation import time_series_cross_validate
from model.features.ema_to_ema_ratio import EmaToEmaRatio

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
pd.set_option('display.max_columns', None)  # Displaying all columns when printing
pd.set_option('display.expand_frame_repr', False)  # Disable line wrap when printing

from data import RAW_DATA_FILE_PATH
from data.raw_data_columns import DataColumns

from config import Config, Environment
from data.convert_to_df import convert_to_data_frame
from data.exchange.exchange_client import ExchangeClient

from datetime import datetime


def train():
    env = Environment.PROD
    print(f'Training using on {env.name} env')

    train_start = datetime(2024, 1, 1)
    train_end = datetime(2025, 2, 3)

    client = BinanceClient(Config(env))
    train_data = get_train_data(client, train_start, train_end)
    time_series_cross_validate(train_data, None)
    for ema in [(20, 8), (15,5), (20, 5), (15, 8)]:
        long, short = ema
        feat = EmaToEmaRatio(long, short)
        time_series_cross_validate(train_data, feat)


def get_train_data(client, train_start, train_end, ) -> pd.DataFrame:
    print(f'Getting train data between {train_start} - {train_end}')
    try:
        df = pd.read_parquet(RAW_DATA_FILE_PATH)
        filtered_by_dates = (df[DataColumns.DATE_CLOSE] >= train_start) & (df[DataColumns.DATE_CLOSE] <= train_end)
        print(f'Got data from existing file')
        return df.loc[filtered_by_dates]
    except FileNotFoundError:
        print(f'Data file not present, downloading')
        train_klines = client.get_historical_klines(ExchangeClient.BTC_USDT_SYMBOL, train_start, train_end)
        train_data = convert_to_data_frame(train_klines)
        train_data.to_parquet(RAW_DATA_FILE_PATH)
    return train_data


if __name__ == "__main__":
    train()
