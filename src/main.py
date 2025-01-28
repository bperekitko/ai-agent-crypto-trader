from config import ENVIRONMENT
from data.exchange.binance.binance_client import BinanceClient
from trading.btc_trader import BtcTrader
from utils.log import get_logger

_LOG = get_logger("Main")

def main():
    client = BinanceClient()
    BtcTrader(client).start()
    client.start()

    # TODO trailing stop loss implementation instead of take profit?
    # TODO info about performed trade in logs? plus minus
    # TODO trades statistics
    # TODO https://alternative.me/crypto/fear-and-greed-index/



if __name__ == "__main__":
    _LOG.info(f'Starting on {ENVIRONMENT} env')
    main()
