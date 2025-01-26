from data.exchange.binance.binance_client import BinanceClient
from trading.btc_trader import BtcTrader

from config import ENVIRONMENT
from utils.log import get_logger

LOG = get_logger("Main")

def main():
    client = BinanceClient()
    BtcTrader(client).start()

    # TODO trailing stop loss implementation instead of take profit?
    # TODO info about performed trade in logs? plus minus
    # TODO trades statistics

    client.start()


if __name__ == "__main__":
    LOG.info(f'Starting on {ENVIRONMENT} env')
    main()
