import os

import pandas as pd

from data.exchange.binance.binance_client import BinanceClient
from model.evaluation.evaluate_binary_model import evaluate_binary_model, evaluate_simple_stats
from model.features.analyze.analyze import analyze
from model.features.analyze.feature_correlation import analyze_correlation
from model.features.bollinger_bands import BollingerBandsWidth
from model.features.close_price_prct_diff import CloseDiff
from model.features.close_to_ema import CloseToEma
from model.features.close_to_low import CloseToLow
from model.features.commodity_channel_index import CommodityChannelIndex
from model.features.ema_to_ema_ratio import EmaToEmaRatio
from model.features.high_to_close import HighToClose
from model.features.hour_of_day import HourOfDaySine, HourOfDayCosine
from model.features.macd import MacdSignal, MacdHistogram, MacdLine
from model.features.rsi import RSI
from model.features.stochastic_oscillator import StochasticOscillator
from model.features.target import HighAboveThreshold, LowAboveThreshold
from model.features.volume import Volume
from model.lstm.binary_lstm import LongHighPriceLstm, LongLowPriceLstm, LongTradeLstm

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
    train_end = datetime(2025, 1, 1)

    client = BinanceClient(Config(env))
    train_data = get_train_data(client, train_start, train_end)
    test_data = get_train_data(client, train_end, datetime(2025, 2, 2))

    model = LongHighPriceLstm()
    model.train(train_data)
    model.test(test_data)




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
