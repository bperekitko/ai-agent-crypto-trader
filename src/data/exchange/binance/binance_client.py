from datetime import datetime
from typing import List

from binance import Client, ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException

from config import Environment, Config
from data.exchange.candlestick import Candlestick
from data.exchange.exchange_client import ExchangeClient
from data.exchange.exchange_error import ExchangeError
from data.exchange.klines_event_listener import KlinesEventListener
from data.exchange.order import Order
from data.exchange.position import Position
from data.exchange.trade import Trade
from utils.log import get_logger


class BinanceClient(ExchangeClient):

    def __init__(self, config: Config):
        self.__LOG = get_logger("Binance Client")
        is_test = config.env == Environment.TEST
        self.client = Client(testnet=is_test, api_key=config.binance_futures_api_key, api_secret=config.binance_futures_secret_key)
        self.twm = ThreadedWebsocketManager(testnet=is_test, api_key=config.binance_futures_api_key, api_secret=config.binance_futures_secret_key)
        self.klines_events_listeners: {str: List[KlinesEventListener]} = {}

    @staticmethod
    def _with_exceptions_handled(func):
        def inner_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except BinanceAPIException as exc:
                raise ExchangeError(exc.message) from exc

        return inner_func

    def add_klines_event_listener(self, listener: KlinesEventListener, symbol: str):
        has_symbol = symbol in self.klines_events_listeners
        self.klines_events_listeners[symbol] = self.klines_events_listeners[symbol] + [listener] if has_symbol else [listener]

    @_with_exceptions_handled
    def place_order(self, order: Order):
        return self.client.futures_create_order(**order.as_params())

    @_with_exceptions_handled
    def place_batch_orders(self, orders: List[Order]):
        batch_orders = [o.as_params() for o in orders]
        return self.client.futures_place_batch_order(batchOrders=batch_orders)

    @_with_exceptions_handled
    def cancel_all_orders(self, symbol: str):
        return self.client.futures_cancel_all_open_orders(symbol=symbol)

    def get_trades(self, symbol, start_time, end_time) -> List[Trade]:
        result = self.client.futures_account_trades(symbol=symbol, start_time=start_time.timestamp(), end_time=end_time.timestamp())
        trades = []
        for trade in result:
            buyer = trade['buyer']
            commission = trade['commission']
            commission_asset = trade['commissionAsset']
            trade_id = trade['id']
            maker = trade['maker']
            order_id = trade['orderId']
            price = trade['price']
            qty = trade['qty']
            quote_qty = trade['quoteQty']
            realized_pnl = trade['realizedPnl']
            side = trade['side']
            position_side = trade['positionSide']
            symbol = trade['symbol']
            time = trade['time']
            trades.append(Trade(buyer, commission, commission_asset, trade_id, maker, order_id, price, qty, quote_qty, realized_pnl, side, position_side, symbol, time))
        return trades

    @_with_exceptions_handled
    def get_current_positions(self, symbol):
        response = self.client.futures_position_information(symbol=symbol)
        return list(map(lambda pos: Position(pos['symbol'], float(pos['positionAmt']), float(pos['entryPrice']), float(pos['breakEvenPrice']), float(pos['markPrice']),
                                             float(pos['unRealizedProfit']), float(pos['liquidationPrice'])), response))

    def start(self):
        self.twm.start()
        for symbol in self.klines_events_listeners.keys():
            self.twm.start_kline_futures_socket(callback=self.handle_kline_message, symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR)
        self.twm.join()

    @_with_exceptions_handled
    def get_last_klines(self, limit, symbol, interval) -> List[Candlestick]:
        klines = self.client.futures_continous_klines(pair=symbol, interval=interval, limit=limit, contractType='PERPETUAL')
        candlesticks = map(
            lambda k: Candlestick(symbol, float(k[1]), float(k[2]), float(k[3]), float(k[4]), True, datetime.fromtimestamp(k[0] / 1000), datetime.fromtimestamp(k[6] / 1000),
                                  float(k[5])), klines)
        return list(candlesticks)

    @_with_exceptions_handled
    def get_historical_klines(self, symbol: str, start: datetime, end: datetime) -> List[Candlestick]:
        klines = self.client.futures_historical_klines(symbol=symbol, interval=self.client.KLINE_INTERVAL_1HOUR, start_str=int(start.timestamp() * 1000),
                                                       end_str=int(end.timestamp() * 1000), limit=None)
        candlesticks = map(
            lambda k: Candlestick(symbol, float(k[1]), float(k[2]), float(k[3]), float(k[4]), True, datetime.fromtimestamp(k[0] / 1000), datetime.fromtimestamp(k[6] / 1000),
                                  float(k[5])), klines)
        return list(candlesticks)

    @_with_exceptions_handled
    def change_leverage(self, leverage: int, symbol: str):
        self.client.futures_change_leverage(symbol=symbol, leverage=leverage)

    @_with_exceptions_handled
    def get_balance(self, asset: str):
        balances = self.client.futures_account_balance()
        asset_balance = next((balance for balance in balances if balance['asset'] == asset), None)
        return float(asset_balance['balance']) if asset_balance is not None else 0

    def handle_kline_message(self, msg):
        if msg['e'] == 'continuous_kline':
            binance_kline_msg_dict = msg['k']
            symbol = msg['ps']
            open_price = binance_kline_msg_dict['o']
            high_price = binance_kline_msg_dict['h']
            low_price = binance_kline_msg_dict['l']
            close_price = binance_kline_msg_dict['c']
            is_closed = binance_kline_msg_dict['x']
            start_date = datetime.fromtimestamp(binance_kline_msg_dict['t'] // 1000)
            end_date = datetime.fromtimestamp(binance_kline_msg_dict['T'] // 1000)
            volume = binance_kline_msg_dict['v']
            candle = Candlestick(symbol, float(open_price), float(high_price), float(low_price), float(close_price), is_closed, start_date, end_date, float(volume))

            for listener in self.klines_events_listeners[symbol]:
                listener.on_event(candle)
