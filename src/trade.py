from data.exchange.binance.binance_client import BinanceClient
from trading.btc_trader import BtcTrader


def start_trading():
    client = BinanceClient()
    BtcTrader(client).start()
    client.start()

    # TODO info about performed trade in logs? plus minus
    # TODO trades statistics
    # TODO https://alternative.me/crypto/fear-and-greed-index/


if __name__ == "__main__":
    start_trading()
