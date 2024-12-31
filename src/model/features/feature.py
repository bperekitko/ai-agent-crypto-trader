from abc import ABC, abstractmethod

from pandas import DataFrame


class Feature(ABC):

    def __init__(self, name):
        self.lag = 0
        self.__name = name

    @abstractmethod
    def _calculate(self, df: DataFrame):
        pass

    def name(self):
        return self.__name

    def calculate(self, df: DataFrame):
        return self._calculate(df).shift(self.lag)

    def lagged(self, lag: int):
        self.lag = lag
        self.__name = f'{self.name()}_lag_{lag}'
        return self
