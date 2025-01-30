import os

from trading import trading_dir_path

os.environ["ENVIRONMENT"] = "PROD"
from datetime import datetime
import pandas as pd
from config import ENVIRONMENT
from data.exchange.binance.binance_client import BinanceClient
from data.exchange.exchange_client import ExchangeClient


def get_stats():
    client = BinanceClient()
    start_time = datetime(2025, 1, 1)
    end_time = datetime(2025, 1, 30)
    symbol = ExchangeClient.BTC_USDT_SYMBOL

    print(f'Getting stats from {start_time} to {end_time} of {symbol}, ENV: {ENVIRONMENT}')
    result = client.get_trades(symbol, start_time, end_time)
    pnl = 0
    fees = 0

    data = []
    for r in result:
        print(f'Trade on {r.time}, side: {r.side}, amount: {r.qty}, profit: {r.realized_pnl}, fees: {r.commission}')
        pnl += float(r.realized_pnl)
        fees += float(r.commission)
        data.append([r.time, r.side, float(r.qty),float(r.realized_pnl), float(r.commission)])

    df = pd.DataFrame(data, columns=['time', 'side', 'quantity', 'PNL', 'fees'])
    df.to_excel(trading_dir_path('trades.xlsx'), index=False)
    print(f'Total trades: {len(result)}')
    print(f'Total PNL :{pnl}, fees: {fees}, total profit: {pnl - fees}')


if __name__ == "__main__":
    get_stats()
