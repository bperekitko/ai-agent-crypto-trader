import threading
import traceback

import numpy as np

from data.convert_to_df import convert_to_data_frame
from data.exchange.candlestick import Candlestick
from data.exchange.exchange_client import ExchangeClient
from data.exchange.klines_event_listener import KlinesEventListener
from data.exchange.order import MarketOrder, OrderSide
from data.raw_data_columns import DataColumns
from model.features.target import TargetLabel
from model.lstm.lstm import Lstm
from utils.log import get_logger

TRADING_THRESHOLD_PROBABILITY = 0.4
LEVERAGE = 4

class BtcTrader(KlinesEventListener):
    def __init__(self, trading_client: ExchangeClient):
        self.model = Lstm.load("0.01")
        self.__LOG = get_logger(f'{ExchangeClient.BTC_USDT_SYMBOL} Trader')
        self.trading_client = trading_client
        self.trading_client.add_klines_event_listener(self, ExchangeClient.BTC_USDT_SYMBOL)
        self.starting_balance = self.trading_client.get_balance("USDT")
        self.__LOG.info(f'Setting up leverage to {LEVERAGE}')
        self.trading_client.change_leverage(LEVERAGE, ExchangeClient.BTC_USDT_SYMBOL)

    def on_event(self, candle: Candlestick):
        if candle.is_closed:
            t1 = threading.Thread(target=self.on_candle_closed, name='Closed Candle Handler', args=[candle])
            t1.start()

    def on_candle_closed(self, candle: Candlestick):
        try:
            signal, probability, target_up, target_down = self.get_trading_signal()
            if signal is None:
                self.__LOG.info('NO SIGNAL RECEIVED')
            else:
                self.__LOG.info(f'TRADING SIGNAL: {signal}, confidence: {probability * 100:.2f}%')
                side = OrderSide.BUY if signal == TargetLabel.UP else OrderSide.SELL
                price = candle.close_price
                target_down_as_percent = abs(target_down / 100)
                target_up_as_percent = abs(target_up / 100)

                self.close_existing_positions()

                trade_quantity = round(candle.close_price / (self.starting_balance / 2), 3)
                new_order = MarketOrder(ExchangeClient.BTC_USDT_SYMBOL, side, trade_quantity)
                stop_loss = new_order.derive_stop_loss(price, target_down_as_percent if side == OrderSide.BUY else target_up_as_percent)
                take_profit = new_order.derive_take_profit(price, target_up_as_percent if side == OrderSide.BUY else target_down_as_percent)

                self.__LOG.info(f'Placing a trade: {side}, price: {price}, stop_loss: {stop_loss.stop_price}, take_profit: {take_profit.stop_price}, {take_profit.price}')
                self.trading_client.place_batch_orders([new_order, stop_loss, take_profit])

        except Exception as error:
            self.__LOG.error("Error while handling closed candle", error)

    def close_existing_positions(self):
        positions = self.trading_client.get_current_positions(self.trading_client.BTC_USDT_SYMBOL)
        if len(positions) > 0:
            self.__LOG.info(f'Found {len(positions)} existing positions - cancelling them')
        for position in positions:
            close_order = position.convert_to_market_close_order()
            self.trading_client.place_order(close_order)
        self.trading_client.cancel_all_orders(self.trading_client.BTC_USDT_SYMBOL)

    def get_trading_signal(self):
        klines = self.trading_client.get_last_klines(20, self.trading_client.BTC_USDT_SYMBOL, '1h')
        input_data = convert_to_data_frame(klines)

        tail = input_data.tail(1)
        start_time = tail[DataColumns.DATE_OPEN].dt.strftime('%Y-%m-%d %H:%M').values[0]
        end_time = tail[DataColumns.DATE_CLOSE].dt.strftime('%Y-%m-%d %H:%M').values[0]
        self.__LOG.info(f'Checking for trading signals for candle: {start_time} - {end_time}')

        predictions, target_up, target_down = self.model.predict(input_data)

        next_interval_predictions = predictions[0]
        self.__LOG.debug(f'Predictions: {[f'{TargetLabel(index).name}:{a_prediction:.4f}' for index, a_prediction in enumerate(next_interval_predictions)]}')

        current_highest_prediction = np.argmax(next_interval_predictions)
        if np.max(next_interval_predictions) < TRADING_THRESHOLD_PROBABILITY:
            self.__LOG.debug(f'There will be no trading my friend, as highest prediction was: {round(next_interval_predictions[current_highest_prediction], 4)}')
            return None, None, None, None
        else:
            return TargetLabel(0), next_interval_predictions[current_highest_prediction], float(target_up), float(target_down)
