from pandas import DataFrame

from data.raw_data_columns import RawDataColumns
from model.features.feature import Feature


class CloseToSma20(Feature):
    __NAME = 'close_to_sma_20'

    def name(self):
        return self.__NAME

    def calculate(self, df: DataFrame):
        sma20 = df[RawDataColumns.CLOSE].rolling(window=20).mean()
        return df[RawDataColumns.CLOSE] / sma20
