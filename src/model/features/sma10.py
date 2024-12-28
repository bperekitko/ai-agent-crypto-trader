from pandas import DataFrame

from data.raw_data_columns import RawDataColumns
from model.features.feature import Feature


class Sma10(Feature):
    __NAME = 'sma_10'

    def name(self):
        return self.__NAME

    def calculate(self, df: DataFrame):
        return df[RawDataColumns.CLOSE].rolling(window=10).mean()
