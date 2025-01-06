import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from data.raw_data_columns import DataColumns
from model.features.analyze.analyze import box_cox_transform
from model.features.feature import Feature


class SharpeRatio(Feature):
    __RISK_FREE_RATE = 0

    def __init__(self, window):
        super().__init__(f'sharpe_ratio_{window}')
        self.__window = window
        self.__scaler = StandardScaler()
        self.__is_fitted = False

    def _calculate(self, data_frame: DataFrame):
        df = data_frame.copy()
        df['returns'] = df[DataColumns.CLOSE].pct_change()
        values = (df['returns'].rolling(window=self.__window).mean() - SharpeRatio.__RISK_FREE_RATE) / df['returns'].rolling(window=self.__window).std()
        values.dropna(inplace=True)

        scaled = self.__scale(values.values.reshape(-1, 1))

        self.__is_fitted = True
        result = pd.Series(scaled.flatten())
        return pd.concat([pd.Series([np.nan] * (len(df) - len(scaled))), result]).reset_index(drop=True)

    def __scale(self, values):
        return self.__scaler.transform(values) if self.__is_fitted else self.__scaler.fit_transform(values)
