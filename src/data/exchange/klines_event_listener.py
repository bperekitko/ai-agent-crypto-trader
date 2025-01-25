from abc import ABC, abstractmethod

from data.exchange.candlestick import Candlestick


class KlinesEventListener(ABC):

    @abstractmethod
    def on_event(self, candle: Candlestick):
        pass

