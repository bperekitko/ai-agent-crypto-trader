from abc import ABC, abstractmethod
from typing import List

from data.exchange.candlestick import Candlestick
from data.exchange.klines_event_listener import KlinesEventListener
from data.exchange.order import Order
from data.exchange.position import Position
from data.exchange.trade import Trade


class ExchangeClient(ABC):
    BTC_USDT_SYMBOL = 'BTCUSDT'

    @abstractmethod
    def add_klines_event_listener(self, listener: KlinesEventListener, symbol: str):
        pass

    @abstractmethod
    def place_order(self, order: Order):
        pass

    @abstractmethod
    def place_batch_orders(self, orders: List[Order]):
        pass

    @abstractmethod
    def cancel_all_orders(self, symbol: str):
        pass

    @abstractmethod
    def get_trades(self) -> List[Trade]:
        pass

    @abstractmethod
    def get_last_klines(self, limit: int, symbol: str, interval: str) -> List[Candlestick]:
        pass

    @abstractmethod
    def get_current_positions(self, symbol: str) -> List[Position]:
        pass

    @abstractmethod
    def change_leverage(self, leverage: int, symbol: str):
        pass

    @abstractmethod
    def get_balance(self, asset: str):
        pass

    @abstractmethod
    def start(self):
        pass
