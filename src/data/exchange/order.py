from abc import ABC, abstractmethod
from enum import Enum


class OrderType(Enum):
    LIMIT = 0,
    MARKET = 1,
    STOP_MARKET = 2,
    TAKE_PROFIT_MARKET = 3,
    TAKE_PROFIT = 4,
    STOP = 5,
    TRAILING_STOP_MARKET = 6


class OrderSide(Enum):
    BUY = 0,
    SELL = 1

    def reversed(self):
        return OrderSide.BUY if self == OrderSide.SELL else OrderSide.SELL


class Order(ABC):
    def __init__(self, symbol, side: OrderSide, order_type: OrderType, price: float, quantity: float):
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.price = price
        self.quantity = quantity

    def derive_stop_loss(self, percent: float):
        stop_loss_price = self.price * (1 - percent) if self.side == OrderSide.BUY else self.price * (1 + percent)
        return StopLossMarketOrder(self.symbol, self.side.reversed(), round(stop_loss_price, 2), round(stop_loss_price, 2))

    def derive_take_profit(self, percent: float):
        take_profit_price = self.price * (1 + percent) if self.side == OrderSide.BUY else self.price * (1 - percent)
        activation_price = self.price * (1 + percent * 0.7) if self.side == OrderSide.BUY else self.price * (1 - percent * 0.7)
        return TakeProfitLimitOrder(self.symbol, self.side.reversed(), round(activation_price, 0), round(take_profit_price, 0), self.quantity)

    @abstractmethod
    def as_params(self):
        pass


class MarketOrder(Order):
    def __init__(self, symbol, side: OrderSide, quantity: float, price: float):
        super().__init__(symbol, side, OrderType.MARKET, price, quantity)

    def as_params(self):
        return {
            'symbol': self.symbol,
            'type': self.order_type.name,
            'side': self.side.name,
            'quantity': f'{self.quantity}'
        }


class StopLimitOrder(Order):
    def __init__(self, symbol: str, side: OrderSide, stop_price: float, price: float, quantity: float):
        super().__init__(symbol, side, OrderType.STOP, price, quantity)
        self.stop_price = stop_price

    def as_params(self):
        return {
            'symbol': self.symbol,
            'type': self.order_type.name,
            'side': self.side.name,
            'quantity': f'{self.quantity}',
            'stopPrice': f'{self.stop_price}',
            'price': f'{self.price}',
            'workingType': 'MARK_PRICE'
        }


class TrailingStopMarketOrder(Order):

    def __init__(self, symbol: str, side: OrderSide, price: float, qty: float, activation_price: float, tolerance: float):
        super().__init__(symbol, side, OrderType.TRAILING_STOP_MARKET, price, qty)
        self.activation_price = activation_price
        self.tolerance = tolerance

    def as_params(self):
        return {
            'symbol': self.symbol,
            'type': self.order_type.name,
            'side': self.side.name,
            'reduceOnly': True,
            'quantity': f'{self.quantity}',
            'callbackRate': f'{self.tolerance}',
            'activationPrice': f'{self.activation_price}',
            'workingType': 'MARK_PRICE'
        }


class StopMarketOrder(Order):
    def __init__(self, symbol, side: OrderSide, stop_price, price, quantity: float):
        super().__init__(symbol, side, OrderType.STOP_MARKET, price, quantity)
        self.stop_price = stop_price

    def as_params(self):
        return {
            'symbol': self.symbol,
            'type': self.order_type.name,
            'side': self.side.name,
            'stopPrice': f'{self.stop_price}',
            'quantity': f'{self.quantity}',
            'workingType': 'MARK_PRICE'
        }

class StopLossMarketOrder(StopMarketOrder):
    def __init__(self, symbol, side: OrderSide, stop_price, price):
        super().__init__(symbol, side, OrderType.STOP_MARKET, price, 0)
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
        super().__init__(symbol, side, OrderType.TAKE_PROFIT, price, quantity)
        self.stop_price = stop_price

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
