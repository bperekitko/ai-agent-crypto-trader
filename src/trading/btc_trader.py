import threading

import numpy as np

from data.convert_to_df import convert_to_data_frame
from data.exchange.candlestick import Candlestick
from data.exchange.exchange_client import ExchangeClient
from data.exchange.exchange_error import ExchangeError
from data.exchange.klines_event_listener import KlinesEventListener
from data.exchange.order import MarketOrder, OrderSide, Order
from data.raw_data_columns import DataColumns
from model.features.target import TargetLabel
from model.lstm.lstm import Lstm
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

    def start(self):
        _LOG.info(f'Starting current balance: {self.starting_balance} $BNFCR')
        _LOG.info(f'Will be using {self.starting_balance * MAX_BALANCE_USED_PER_POSITION} $BNFCR per position')

        _LOG.info(f'Setting up leverage to {LEVERAGE}')
        self.trading_client.change_leverage(LEVERAGE, ExchangeClient.BTC_USDT_SYMBOL)
        _LOG.info(f'Leverage set, starting listening to klines')
        self.trading_client.add_klines_event_listener(self, ExchangeClient.BTC_USDT_SYMBOL)
        _LOG.info(f'Trader initialized, thank you')

    def on_event(self, candle: Candlestick):
        # _LOG.debug(f'Candle event, price: {candle.close_price}, closed?: {candle.is_closed}')
        if candle.is_closed:
            t1 = threading.Thread(target=self.on_candle_closed, name='BTCUSDT Trader', args=[candle])
            t1.start()

    def on_candle_closed(self, candle: Candlestick):
        try:
            signal, probability, target_up, target_down = self.get_trading_signal()
            if signal == TargetLabel.NEUTRAL or probability < TRADING_THRESHOLD_PROBABILITY:
                _LOG.info(f'NO TRADING: signal {signal.name}, confidence: {probability * 100:.2f}%')
            else:
                _LOG.info(f'TRADING SIGNAL: {signal}, confidence: {probability * 100:.2f}%')
                side = OrderSide.BUY if signal == TargetLabel.UP else OrderSide.SELL
                current_price = candle.close_price

                self.__perform_trade(current_price, side, target_down, target_up)

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

            new_order = MarketOrder(ExchangeClient.BTC_USDT_SYMBOL, side, round(trade_quantity, 3), current_price)
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
        new_order = MarketOrder(ExchangeClient.BTC_USDT_SYMBOL, side, quantity, current_price)
        stop_loss = new_order.derive_stop_loss(target_down if side == OrderSide.BUY else target_up)
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
        klines = self.trading_client.get_last_klines(20, self.trading_client.BTC_USDT_SYMBOL, '1h')
        input_data = convert_to_data_frame(klines)

        tail = input_data.tail(1)
        start_time = tail[DataColumns.DATE_OPEN].dt.strftime('%Y-%m-%d %H:%M').values[0]
        end_time = tail[DataColumns.DATE_CLOSE].dt.strftime('%Y-%m-%d %H:%M').values[0]
        _LOG.info(f'Checking for trading signals for candle: {start_time} - {end_time}')

        predictions, target_up, target_down = self.model.predict(input_data)

        next_interval_predictions = predictions[0]
        _LOG.debug(f'Predictions: {[f'{TargetLabel(index).name}:{a_prediction:.4f}' for index, a_prediction in enumerate(next_interval_predictions)]}')

        current_highest_prediction = np.argmax(next_interval_predictions)

        target_down_as_percent = abs(float(target_down) / 100)
        target_up_as_percent = abs(float(target_up) / 100)
        return TargetLabel(current_highest_prediction), next_interval_predictions[current_highest_prediction], target_up_as_percent, target_down_as_percent
