import pandas as pd
import numpy as np
from data.raw_data_columns import DataColumns
from model.features.analyze.analyze import box_cox_transform
from model.features.feature import Feature
from sklearn.preprocessing import StandardScaler


class VolumeRatio(Feature):
    def __init__(self, period=20):
        super().__init__(f'volume_ratio_{period}')
        self.period = period
        self.scaler = StandardScaler()
        self.fitted_lambda = None
        self.box_cox_shift = None
        self.is_fitted = False

    def _calculate(self, input_df: pd.DataFrame):
        df = input_df.copy()
        df['avg_volume'] = df[DataColumns.VOLUME].rolling(window=self.period).mean()
        series = df[DataColumns.VOLUME] / df['avg_volume']
        series = self.__transform(series.dropna())
        if self._bins > 0:
            series = self._binned_equal_size(series)

        scaled = self.__scale_values(series.reshape(-1, 1))
        self.is_fitted = True

        result= pd.Series(scaled.flatten())

        len_diff = len(df) - len(scaled)
        return pd.concat([pd.Series([np.nan] * len_diff), result]).reset_index(drop=True) if len_diff > 0 else result

    def __scale_values(self, values):
        return self.scaler.transform(values) if self.is_fitted else self.scaler.fit_transform(values)

    def __transform(self, series: pd.Series):
        if self.is_fitted:
            return box_cox_transform(series, self.fitted_lambda, self.box_cox_shift)
        else:
            result, self.fitted_lambda, self.box_cox_shift = box_cox_transform(series)
            return result
