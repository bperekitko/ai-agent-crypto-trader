import os

import pandas as pd

from data.exchange.binance.binance_client import BinanceClient
from model.evaluation.rolling_window_validation import validate_using_rolling_window
from model.features.analyze.feature_correlation import analyze_correlation
from model.features.analyze.feature_importance import analyze_importance
from model.features.target import LongTradeTarget
from model.lstm.binary_lstm import LongTradeLstm
from model.lstm.hiperparameter_tuning import tune_lstm
from trading.backtest_strategy import backtest_model

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

    all_data = get_train_data(client, train_start, datetime(2025, 2, 2))

    # tune_lstm(train_data)
    # input_df = LongTradeLstm().prepare_data(all_data)
    # analyze_importance(input_df.drop([DataColumns.DATE_CLOSE, LongTradeTarget().name()], axis=1), input_df[LongTradeTarget().name()])
    #
    model = LongTradeLstm()
    model.train(train_data)
    backtest_model(test_data, model, 0.5)
    model.test(test_data)

    # validate_using_rolling_window(get_train_data(client, train_start, datetime(2025, 2, 2)), LongTradeLstm, 12)
    # analyze_correlation(LongTradeLstm().prepare_data(train_data).drop([DataColumns.DATE_CLOSE, LongTradeTarget().name()], axis=1))


def get_train_data(client, train_start, train_end) -> pd.DataFrame:
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
