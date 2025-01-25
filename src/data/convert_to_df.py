from data.exchange.candlestick import Candlestick
from typing import List

import pandas as pd

from data.raw_data_columns import DataColumns


def convert_to_data_frame(candlesticks: List[Candlestick]) -> pd.DataFrame:
    data = map(__candle_as_pd_entry, candlesticks)
    return pd.DataFrame(
        data,
        columns=[
            DataColumns.DATE_OPEN,
            DataColumns.DATE_CLOSE,
            DataColumns.LOW,
            DataColumns.HIGH,
            DataColumns.OPEN,
            DataColumns.CLOSE,
            DataColumns.VOLUME
        ]
    )

def __candle_as_pd_entry(candle: Candlestick) -> List:
    return [candle.start_date, candle.end_date, candle.low_price, candle.high_price, candle.open_price, candle.close_price, candle.volume]
