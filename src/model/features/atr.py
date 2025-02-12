import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from data.raw_data_columns import DataColumns
from model.features.analyze.analyze import box_cox_transform
from model.features.feature import Feature


class AverageTrueRange(Feature):
    def __init__(self, window):
        super().__init__(f'ATR_{window}')
        self.__window = window
        self.scaler = StandardScaler()
        self.fitted_lambda = None
        self.box_cox_shift = None
        self.is_fitted = False

    def _calculate(self, input_df: DataFrame):
        df = input_df.copy()
        df['prev_close'] = df[DataColumns.CLOSE].shift(1)
        df['tr'] = np.maximum(df[DataColumns.HIGH] - df[DataColumns.LOW],
                              np.maximum(abs(df[DataColumns.HIGH] - df['prev_close']),
                                         abs(df[DataColumns.LOW] - df['prev_close'])))

        values = df['tr'].rolling(window=self.__window).mean()

        values.dropna(inplace=True)
        transformed = self.__transform(values)
        scaled = self.__scale(transformed)
        self.is_fitted = True
        result = pd.Series(scaled.flatten())
        return pd.concat([pd.Series([np.nan] * (len(df) - len(scaled))), result]).reset_index(drop=True)

    def __scale(self, values):
        return self.scaler.transform(values) if self.is_fitted else self.scaler.fit_transform(values)

    def __transform(self, series: pd.Series):
        if self.is_fitted:
            return box_cox_transform(series, self.fitted_lambda, self.box_cox_shift).reshape(-1, 1)
        else:
            result, self.fitted_lambda, self.box_cox_shift = box_cox_transform(series)
            return result.reshape(-1, 1)
