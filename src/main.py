from data.exchange.binance.binance_client import BinanceClient
from trading.btc_trader import BtcTrader


def main():
    client = BinanceClient()
    trader = BtcTrader(client)
    client.start()


if __name__ == "__main__":
    main()
