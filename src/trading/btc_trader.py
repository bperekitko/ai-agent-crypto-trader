import threading
from datetime import timedelta

import numpy as np
import pandas as pd

from config import ENVIRONMENT
from data.convert_to_df import convert_to_data_frame
from data.exchange.candlestick import Candlestick
from data.exchange.exchange_client import ExchangeClient
from data.exchange.exchange_error import ExchangeError
from data.exchange.klines_event_listener import KlinesEventListener
from data.exchange.order import MarketOrder, OrderSide, Order, StopLimitOrder
from model.features.target import TargetLabel
from model.lstm.lstm import Lstm
from utils.deque import Dequeue
from utils.log import get_logger

MAX_BALANCE_USED_PER_POSITION = 0.5
TRADING_THRESHOLD_PROBABILITY = 0.45
LEVERAGE = 4
MIN_QTY = 0.001
_LOG = get_logger(f'{ExchangeClient.BTC_USDT_SYMBOL} Trader')


class BtcTrader(KlinesEventListener):

    def __init__(self, trading_client: ExchangeClient):
        self.model = Lstm.load("0.01")
        self.trading_client = trading_client
        self.starting_balance = self.trading_client.get_balance("BNFCR")
        self.predictions_klines = Dequeue[Candlestick](19)
        last_18 = self.trading_client.get_last_klines(19, ExchangeClient.BTC_USDT_SYMBOL, '1h')[:-1]
        for kline in last_18:
            self.predictions_klines.push(kline)

    def start(self):
        _LOG.info(f'Starting on {ENVIRONMENT} env')
        _LOG.info(f'Starting current balance: {self.starting_balance} $BNFCR')
        _LOG.info(f'Will be using {self.starting_balance * MAX_BALANCE_USED_PER_POSITION} $BNFCR per position')

        _LOG.info(f'Setting up leverage to {LEVERAGE}')
        self.trading_client.change_leverage(LEVERAGE, ExchangeClient.BTC_USDT_SYMBOL)
        _LOG.info(f'Leverage set, starting listening to klines')
        self.trading_client.add_klines_event_listener(self, ExchangeClient.BTC_USDT_SYMBOL)
        _LOG.info(f'Trader initialized, thank you')
        _LOG.info(
            f'Waiting for current candle to finish at {(self.predictions_klines.elements[17].end_date + timedelta(hours=1)).strftime('%H:%M')} , next prediction at {(self.predictions_klines.elements[17].start_date + timedelta(hours=2)).strftime('%H:%M')}')

    def on_event(self, candle: Candlestick):
        # _LOG.debug(f'Candle event, price: {candle.close_price}, closed?: {candle.is_closed}')
        if candle.is_closed:
            t1 = threading.Thread(target=self.on_candle_closed, name='BTCUSDT Trader', args=[candle])
            t1.start()

    def on_candle_closed(self, candle: Candlestick):
        try:
            self.predictions_klines.push(candle)
            signal, probability, target_up, target_down = self.get_trading_signal()
            if signal == TargetLabel.NEUTRAL or probability < TRADING_THRESHOLD_PROBABILITY:
                _LOG.info(f'NO TRADING: signal {signal.name}, confidence: {probability * 100:.2f}%')
            else:
                _LOG.info(f'TRADING SIGNAL: {signal}, confidence: {probability * 100:.2f}%')
                side = OrderSide.BUY if signal == TargetLabel.UP else OrderSide.SELL
                current_price = candle.close_price

                self.__perform_trade(current_price, side, target_down, target_up)
            _LOG.info(f'Next prediction at {(self.predictions_klines.elements[self.predictions_klines.size - 1].start_date + timedelta(hours=2)).strftime('%H:%M')}')

        except Exception as error:
            _LOG.exception(error)

    def __perform_trade(self, current_price, side, target_down, target_up):
        existing_positions = self.trading_client.get_current_positions(self.trading_client.BTC_USDT_SYMBOL)
        if len(existing_positions) > 1:
            raise Exception("How come we have more than one position in BTCUSDT?")

        if len(existing_positions) == 1:
            position = existing_positions[0]
            self.__handle_existing_position(position, current_price, side, target_down, target_up)
        else:
            _LOG.info(f'There are no existing positions, creating new order')
            trade_quantity = self.starting_balance * MAX_BALANCE_USED_PER_POSITION * LEVERAGE / current_price
            self.__new_trade(current_price, side, target_down, target_up, trade_quantity)

    def __handle_existing_position(self, position, current_price, side, target_down, target_up):
        if position.side == side:
            _LOG.info(f'Current signal is the same as the existing position, keeping it')
        else:
            _LOG.info(f'Current signal is {side.name}, however {position.side.name} position exists, preparing reversed order')
            self.trading_client.cancel_all_orders(self.trading_client.BTC_USDT_SYMBOL)
            trade_quantity = position.position_amount * 2

            price_activation_threshold = 0.0005
            order_price_activation = (1 + price_activation_threshold) * current_price if side == OrderSide.BUY else (1 - price_activation_threshold) * current_price

            new_order = StopLimitOrder(ExchangeClient.BTC_USDT_SYMBOL, side, round(order_price_activation, 0), round(order_price_activation, 0), round(trade_quantity, 3))
            stop_loss = new_order.derive_stop_loss(target_down if side == OrderSide.BUY else target_up)
            take_profit = new_order.derive_take_profit(target_up if side == OrderSide.BUY else target_down)
            take_profit.quantity = position.position_amount

            _LOG.info(f'Placing a trade: {side}, price: {current_price}, quantity: {trade_quantity},  stop_loss: {stop_loss.stop_price}, take_profit: {take_profit.price}')
            self.__place_order(take_profit)
            self.__place_order(stop_loss)
            self.__place_order(new_order)

    def __new_trade(self, current_price, side, target_down, target_up, trade_quantity):
        self.trading_client.cancel_all_orders(ExchangeClient.BTC_USDT_SYMBOL)

        quantity = round(trade_quantity, 3) if round(trade_quantity, 3) >= MIN_QTY else MIN_QTY

        price_activation_threshold = 0.0005
        order_price_activation = (1 + price_activation_threshold) * current_price if side == OrderSide.BUY else (1 - price_activation_threshold) * current_price

        new_order = StopLimitOrder(ExchangeClient.BTC_USDT_SYMBOL, side, round(order_price_activation, 0), round(order_price_activation, 0), quantity)
        stop_loss = new_order.derive_stop_loss(target_up if side == OrderSide.BUY else target_down)
        take_profit = new_order.derive_take_profit(target_up if side == OrderSide.BUY else target_down)
        _LOG.info(f'Placing a trade: {side}, price: {current_price}, quantity: {quantity},  stop_loss: {stop_loss.stop_price}, take_profit: {take_profit.price}')

        self.__place_order(take_profit)
        self.__place_order(stop_loss)
        self.__place_order(new_order)

    def __place_order(self, order: Order):
        try:
            self.trading_client.place_order(order)
        except ExchangeError as error:
            self.trading_client.cancel_all_orders(ExchangeClient.BTC_USDT_SYMBOL)
            _LOG.error(f'Cannot place {order.order_type.name} order: {error.message}')
            raise error

    def get_trading_signal(self):
        klines = self.predictions_klines.get_all()
        duplicate_last = klines.copy()[self.predictions_klines.size - 1]
        input_data = convert_to_data_frame(klines)
        new_row = convert_to_data_frame([duplicate_last])
        input_for_predict = pd.concat([input_data, new_row], ignore_index=True)

        start_time = (duplicate_last.start_date + timedelta(hours=1)).strftime('%H:%M')
        end_time = (duplicate_last.end_date + timedelta(hours=1)).strftime('%H:%M')

        _LOG.info(f'Checking for trading signals for candle: {start_time} - {end_time}')

        predictions, target_up, target_down = self.model.predict(input_for_predict)

        next_interval_predictions = predictions[0]
        predictions_description = [f'{TargetLabel(index).name}:{a_prediction:.4f}' for index, a_prediction in enumerate(next_interval_predictions)]
        _LOG.debug(f'Predictions: {predictions_description}')

        current_highest_prediction = np.argmax(next_interval_predictions)

        target_down_as_percent = abs(float(target_down) / 100)
        target_up_as_percent = abs(float(target_up) / 100)
        return TargetLabel(current_highest_prediction), next_interval_predictions[current_highest_prediction], target_up_as_percent, target_down_as_percent
