from abc import ABC, abstractmethod
from enum import Enum


class OrderType(Enum):
    LIMIT = 0,
    MARKET = 1,
    STOP_MARKET = 2,
    TAKE_PROFIT_MARKET = 3,
    TAKE_PROFIT = 4,


class OrderSide(Enum):
    BUY = 0,
    SELL = 1

    def reversed(self):
        return OrderSide.BUY if self == OrderSide.SELL else OrderSide.SELL


class Order(ABC):
    def __init__(self, symbol, side: OrderSide, order_type: OrderType):
        self.symbol = symbol
        self.side = side
        self.order_type = order_type

    @abstractmethod
    def as_params(self):
        pass


class MarketOrder(Order):
    def __init__(self, symbol, side: OrderSide, quantity):
        super().__init__(symbol, side, OrderType.MARKET)
        self.quantity = quantity

    def derive_stop_loss(self, entry_price: float, percent: float):
        stop_loss_price = entry_price * (1 - percent) if self.side == OrderSide.BUY else entry_price * (1 + percent)
        return StopLossMarketOrder(self.symbol, self.side.reversed(), round(stop_loss_price, 2))

    def derive_take_profit(self, entry_price: float, percent: float):
        take_profit_price = entry_price * (1 + percent) if self.side == OrderSide.BUY else entry_price * (1 - percent)
        stop_price = entry_price * (1 + percent * 0.7) if self.side == OrderSide.BUY else entry_price * (1 - percent * 0.7)
        return TakeProfitLimitOrder(self.symbol, self.side.reversed(), round(stop_price, 0), round(take_profit_price, 0), self.quantity)

    def as_params(self):
        return {
            'symbol': self.symbol,
            'type': self.order_type.name,
            'side': self.side.name,
            'quantity': f'{self.quantity}'
        }


class StopLossMarketOrder(Order):
    def __init__(self, symbol, side: OrderSide, stop_price):
        super().__init__(symbol, side, OrderType.STOP_MARKET)
        self.stop_price = stop_price

    def as_params(self):
        return {
            'symbol': self.symbol,
            'type': self.order_type.name,
            'side': self.side.name,
            'closePosition': "true",
            'stopPrice': f'{self.stop_price}',
            'workingType': 'MARK_PRICE'
        }


class TakeProfitLimitOrder(Order):
    def __init__(self, symbol, side: OrderSide, stop_price: float, price: float, quantity: float):
        super().__init__(symbol, side, OrderType.TAKE_PROFIT)
        self.stop_price = stop_price
        self.price = price
        self.quantity = quantity

    def as_params(self):
        return {
            'symbol': self.symbol,
            'type': self.order_type.name,
            'side': self.side.name,
            'quantity': f'{self.quantity}',
            'stopPrice': f'{self.stop_price}',
            'price': f'{self.price}',
            'reduceOnly': "true",
            'workingType': 'MARK_PRICE'
        }
