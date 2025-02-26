import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config, Environment
from data import path_for_data
from data.convert_to_df import convert_to_data_frame
from data.exchange.binance.binance_client import BinanceClient
from data.exchange.exchange_client import ExchangeClient
from data.raw_data_columns import DataColumns
from model.model import Model
from trading import trading_dir_path

transaction_cost = 0.001
take_profit = 0.005
stop_loss = 0.005


def prepare_minute_intervals_data(df: pd.DataFrame):
    start = df.head(1)[DataColumns.DATE_OPEN].values[0]
    end = df.tail(1)[DataColumns.DATE_OPEN].values[0]

    start = pd.Timestamp(start).to_pydatetime()
    end = pd.Timestamp(end).to_pydatetime()

    interval = '1m'

    client = BinanceClient(Config(Environment.PROD))
    path = path_for_data(interval, start, end)
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        print(f'Data file not present, downloading')
        train_klines = client.get_historical_klines(ExchangeClient.BTC_USDT_SYMBOL, start, end, interval=interval)
        data_frame = convert_to_data_frame(train_klines)
        data_frame.to_parquet(path)
        return data_frame


def simulate_trade(entry_price, minute_df):
    outcome_fee = None
    outcome_no_fee = None

    target_price = entry_price * (1 + take_profit)
    stop_price = entry_price * (1 - stop_loss)

    for idx, row in minute_df.iterrows():
        reached_target = row[DataColumns.HIGH] >= target_price
        reached_stop = row[DataColumns.LOW] <= stop_price

        if reached_target and reached_stop:
            outcome_no_fee = -stop_loss
            outcome_fee = outcome_no_fee - transaction_cost
            break
        elif reached_target:
            outcome_no_fee = take_profit
            outcome_fee = outcome_no_fee - transaction_cost
            break
        elif reached_stop:
            outcome_no_fee = -stop_loss
            outcome_fee = outcome_no_fee - transaction_cost
            break

    if outcome_no_fee is None:
        final_return = (minute_df.iloc[-1][DataColumns.CLOSE] / entry_price) - 1
        outcome_no_fee = final_return
        outcome_fee = final_return - transaction_cost

    return outcome_fee, outcome_no_fee


def compute_stats(returns_list):
    if returns_list:
        cumulative = np.prod([1 + r for r in returns_list]) - 1
        avg_return = np.mean(returns_list)
        win_rate = np.mean(np.array(returns_list) > 0)
        return {
            "n_trades": len(returns_list),
            "avg_return": f'{round(avg_return * 100, 2)}%',
            "win_rate":  f'{round(win_rate * 100, 2)}%',
            "cumulative_return":  f'{round(cumulative * 100, 2)}%',
        }
    else:
        return {
            "n_trades": 0,
            "avg_return": None,
            "win_rate": None,
            "cumulative_return": None
        }


def backtest_model(input_df: pd.DataFrame, trained_model: Model, prediction_confidence_threshold: float):
    df = input_df.copy()
    df = df.reset_index(drop=True)

    minute_data = prepare_minute_intervals_data(df)
    preds, y_true = trained_model.predict(pd.DataFrame(df))
    preds = preds.flatten()

    offset = len(df) - len(preds)
    df['predicted_target'] = np.nan
    df.loc[df.index[offset:], 'predicted_target'] = np.where(preds > prediction_confidence_threshold, 1, 0)

    trades_with_fees = []
    trades_no_fees = []
    trades_dates = []

    missed_with_fees = []
    missed_no_fees = []
    missed_dates = []

    for i in range(len(df) - 1):
        entry_price = df.loc[i + 1, DataColumns.OPEN]

        filtered_by_dates = (
                (minute_data[DataColumns.DATE_CLOSE] >= df.loc[i + 1, DataColumns.DATE_OPEN].to_pydatetime())
                & (minute_data[DataColumns.DATE_CLOSE] <= df.loc[i + 1, DataColumns.DATE_CLOSE].to_pydatetime())
        )

        minute_df = minute_data.loc[filtered_by_dates]

        outcome_fee, outcome_no_fee = simulate_trade(entry_price, minute_df)

        if df.loc[i, 'predicted_target'] == 1:
            trades_with_fees.append(outcome_fee)
            trades_no_fees.append(outcome_no_fee)
            trades_dates.append(df.loc[i + 1, DataColumns.DATE_OPEN])
        else:
            if outcome_no_fee >= take_profit:
                missed_with_fees.append(outcome_fee)
                missed_no_fees.append(outcome_no_fee)
                missed_dates.append(df.loc[i + 1, DataColumns.DATE_OPEN])

    stats_trades_with_fees = compute_stats(trades_with_fees)
    stats_trades_no_fees = compute_stats(trades_no_fees)
    stats_missed_with_fees = compute_stats(missed_with_fees)
    stats_missed_no_fees = compute_stats(missed_no_fees)

    print("Trades made: ")
    print("  With fees: ", stats_trades_with_fees)
    print("  Without fees: ", stats_trades_no_fees)

    print("\nMissed trades:")
    print("  With fees: ", stats_missed_with_fees)
    print("  Without fees: ", stats_missed_no_fees)

    if trades_with_fees:
        cum_returns_trades = np.cumprod([1 + r for r in trades_with_fees]) - 1
        plt.figure(figsize=(10, 5))
        plt.plot(trades_dates, cum_returns_trades, marker='o', linestyle='-', label="Trades (with fees)")
        plt.title("Cumulative return - performed trades")
        plt.xlabel("Data")
        plt.ylabel("Cumulative return")
        plt.legend()
        plt.grid(True)
        plt.savefig(trading_dir_path("backtest_result_trades.png"))

    stats = {
        "trades": {
            "with_fees": stats_trades_with_fees,
            "no_fees": stats_trades_no_fees
        },
        "missed_trades": {
            "with_fees": stats_missed_with_fees,
            "no_fees": stats_missed_no_fees
        }
    }

    with open(trading_dir_path("backtest_stats.json"), "w") as file:
        json.dump(stats, file, indent=4, default=str)

    print("Stats save to file backtest_stats.json")
