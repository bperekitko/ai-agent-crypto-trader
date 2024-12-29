from pandas import DataFrame

from data.raw_data_columns import DataColumns
from model.features.feature import Feature


class CloseToLow(Feature):
    __NAME = 'close_to_low_percent'

    def name(self):
        return self.__NAME

    def calculate(self, df: DataFrame):
        return ((df[DataColumns.CLOSE] - df[DataColumns.LOW]) / df[DataColumns.LOW]) * 100
