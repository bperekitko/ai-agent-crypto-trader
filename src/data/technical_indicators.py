import pandas as pd
import numpy as np
import ta
import ta.momentum

from data.raw_data_columns import DataColumns


def add_technical_indicators_to_data(file_path):
    df = pd.read_parquet(file_path)
    # Oblicz RSI
    df['rsi_14'] = ta.momentum.RSIIndicator(df[DataColumns.CLOSE], window=14).rsi()
    df['rsi_9'] = ta.momentum.RSIIndicator(df[DataColumns.CLOSE], window=9).rsi()
    df['rsi_25'] = ta.momentum.RSIIndicator(df[DataColumns.CLOSE], window=25).rsi()

    # Oblicz MACD
    macd = ta.trend.MACD(df[DataColumns.CLOSE])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # Oblicz Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df[DataColumns.CLOSE], window=20, window_dev=2)
    df['bollinger_mavg'] = bollinger.bollinger_mavg()
    df['bollinger_hband'] = bollinger.bollinger_hband()
    df['bollinger_lband'] = bollinger.bollinger_lband()

    df['volatility_24'] = df[DataColumns.CLOSE].rolling(window=24).std()
    df['volatility_12'] = df[DataColumns.CLOSE].rolling(window=12).std()

    df['cum_vol_24'] = df[DataColumns.VOLUME].rolling(window=24).sum()
    df['cum_vol_12'] = df[DataColumns.VOLUME].rolling(window=12).sum()

    df['close_diff'] = df['close'].diff()
    # df['percent_change'] = (df['close_diff'] / (df['close'] - df['close_diff'])) * 100

    df['date_open'] = pd.to_datetime(df['date_open'])
    df['date_close'] = pd.to_datetime(df['date_close'])
    df['hour_sin'] = np.sin(2 * np.pi * df['date_open'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['date_open'].dt.hour / 24)
    df['day_of_week'] = df['date_open'].dt.dayofweek
    df = pd.get_dummies(df, columns=['day_of_week'], prefix='weekday', dtype=int)

    df.to_parquet(file_path)


def add_target_to_data(file_path):
    df = pd.read_parquet(file_path)
    df[DataColumns.CLOSE_PERCENT_CHANGE] = np.round(df[DataColumns.CLOSE].pct_change() * 100, 5)
    positive_changes = df[df[DataColumns.CLOSE_PERCENT_CHANGE] > 0][DataColumns.CLOSE_PERCENT_CHANGE]
    negative_changes = df[df[DataColumns.CLOSE_PERCENT_CHANGE] < 0][DataColumns.CLOSE_PERCENT_CHANGE]

    positive_percentiles = positive_changes.describe(percentiles=[0.33])
    negative_percentiles = negative_changes.describe(percentiles=[0.66])
    threshold_up = positive_percentiles.loc['33%']
    threshold_down = negative_percentiles.loc['66%']

    print('UP target threshold: ', threshold_up)
    print('DOWN target threshold: ', threshold_down)

    def __calculate_target(row, upper_threshold, lower_threshold):
        if row < lower_threshold:
            return "DOWN"
        elif lower_threshold <= row <= upper_threshold:
            return "NEUTRAL"
        else:
            return "UP"

    df[DataColumns.TARGET] = df[DataColumns.CLOSE_PERCENT_CHANGE].apply(
        lambda x: __calculate_target(x, threshold_up, threshold_down))
    df[DataColumns.TARGET] = df[DataColumns.TARGET].shift(-1)
    print(df[DataColumns.TARGET].describe(percentiles=[0.33, 0.66]))
    print(df[DataColumns.TARGET].value_counts())
    df.to_parquet(file_path)
