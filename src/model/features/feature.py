from abc import ABC, abstractmethod

import pandas as pd


class Feature(ABC):

    def __init__(self, name):
        self.lag = 0
        self._bins = 0
        self.__name = name

    @abstractmethod
    def _calculate(self, df: pd.DataFrame):
        pass

    def name(self):
        return self.__name

    def calculate(self, df: pd.DataFrame):
        return self._calculate(df).shift(self.lag)

    def lagged(self, lag: int):
        self.lag = lag
        self.__name = f'{self.name()}_lag_{lag}'
        return self

    def binned_equally(self, bins_num: int):
        self._bins = bins_num
        self.__name = f'{self.name()}_binned_equally_{bins_num}'
        return self

    def _binned_equal_size(self, values):
        return pd.cut(values, bins=self._bins, labels=False, include_lowest=True)
