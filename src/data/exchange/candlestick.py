import json


class Candlestick:
    def __init__(self, symbol: str, open_price: float, high_price: float, low_price: float, close_price: float, is_closed: bool, start_date, end_date, volume: float):
        self.symbol = symbol
        self.open_price = open_price
        self.high_price = high_price
        self.low_price = low_price
        self.close_price = close_price
        self.is_closed = is_closed
        self.start_date = start_date
        self.end_date = end_date
        self.volume = volume

    def __str__(self):
        return json.dumps(
            {
                "symbol": self.symbol,
                "open_price": self.open_price,
                "high_price": self.high_price,
                "low_price": self.low_price,
                "close_price": self.close_price,
                "is_closed": self.is_closed,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "volume": self.volume
            }, default=str
        )
