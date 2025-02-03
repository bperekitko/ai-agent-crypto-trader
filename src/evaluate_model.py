import os

from config import Config, Environment
from data.raw_data_columns import DataColumns
from model.lstm.lstm import Lstm

from datetime import datetime
import pandas as pd
from data.exchange.binance.binance_client import BinanceClient
from data.exchange.exchange_client import ExchangeClient


def evaluate_model():
    env = Environment.PROD
    print(f'Starting on {env.name} env')
    client = BinanceClient(Config(env))
    start_time = datetime(2024, 12, 31)
    end_time = datetime(2025, 1, 31)
    symbol = ExchangeClient.BTC_USDT_SYMBOL
    print(f'Getting historical klines between {start_time} and {end_time}')
    result = client.get_historical_klines(symbol, start_time, end_time)
    data = []
    print(f'Got {len(result)} candles, preparing DataFrame')
    for candle in result:
        data.append([candle.start_date, candle.open_price, candle.high_price, candle.low_price, candle.close_price, candle.volume, candle.end_date])
    df = pd.DataFrame(
        data,
        columns=[
            DataColumns.DATE_OPEN,
            DataColumns.OPEN,
            DataColumns.HIGH,
            DataColumns.LOW,
            DataColumns.CLOSE,
            DataColumns.VOLUME,
            DataColumns.DATE_CLOSE,
        ],
    )

    model = Lstm.load('0.01')
    print(f'Model loaded')
    model.test(df)


if __name__ == "__main__":
    evaluate_model()
