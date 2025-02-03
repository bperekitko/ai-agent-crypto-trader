import os

from config import Environment, Config
from data.exchange.binance.binance_client import BinanceClient
from trading.btc_trader import BtcTrader
from utils.log import get_logger

_LOG = get_logger("MainApp")
def start_trading():
    config = Config(Environment.PROD if os.getenv("ENVIRONMENT") == 'PROD' else Environment.TEST)
    _LOG.info(f'Starting on {config.env.name} env')

    client = BinanceClient(config)
    BtcTrader(client).start()
    client.start()

    # TODO info about performed trade in logs? plus minus
    # TODO trades statistics
    # TODO https://alternative.me/crypto/fear-and-greed-index/


if __name__ == "__main__":
    start_trading()
