from abc import ABC, abstractmethod

import pandas as pd


class Model(ABC):
    def __init__(self):
        self.params = {}

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def version(self) -> str:
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def train(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def test(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self, version: str):
        pass
