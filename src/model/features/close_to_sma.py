import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from data.raw_data_columns import DataColumns
from model.features.feature import Feature


class CloseToSma(Feature):

    def __init__(self, window=20):
        super().__init__(f'close_to_sma_{window}')
        self.scaler = StandardScaler()
        self.window = window
        self.fitted_lambda = None
        self.box_cox_shift = None
        self.is_fitted = False


    def _calculate(self, df: DataFrame):
        sma = df[DataColumns.CLOSE].rolling(window=self.window).mean()
        series = df[DataColumns.CLOSE] / sma
        series.dropna(inplace=True)

        scaled_values = self.__scale_values(series.values.reshape(-1, 1))
        self.is_fitted = True

        result = pd.Series(scaled_values.flatten())
        return pd.concat([pd.Series([np.nan] * (len(df) - len(scaled_values))), result]).reset_index(drop=True)

    def __scale_values(self, values: pd.Series):
        transformation_func = self.scaler.transform if self.is_fitted else self.scaler.fit_transform
        return transformation_func(values)
