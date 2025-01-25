import json

from data.exchange.order import OrderSide, Order, MarketOrder


class Position:
    def __init__(self, symbol: str, position_amount: float, entry_price: float, break_even_price: float, mark_price: float, un_realized_profit: float, liquidation_price: float):
        self.symbol = symbol
        self.position_amount = position_amount
        self.side = OrderSide.BUY if position_amount > 0 else OrderSide.SELL
        self.entry_price = entry_price
        self.break_even_price = break_even_price
        self.mark_price = mark_price
        self.un_realized_profit = un_realized_profit
        self.liquidation_price = liquidation_price

    def convert_to_market_close_order(self) -> Order:
        closing_side = self.side.reversed()
        qty = abs(self.position_amount)
        return MarketOrder(self.symbol, side=closing_side, quantity=qty)

    def __str__(self):
        return json.dumps(
            {
                "symbol": self.symbol,
                "position_amount": self.position_amount,
                "side": self.side.name,
                "entry_price": self.entry_price,
                "break_even_price": self.break_even_price,
                "mark_price": self.mark_price,
                "un_realized_profit": self.un_realized_profit,
                "liquidation_price": self.liquidation_price
            }
        )
