from model.features.close_price_prct_diff import CloseDiff
from model.features.close_to_low import CloseToLow
from model.features.high_to_close import HighToClose
from model.features.close_to_sma20 import CloseToSma20


def get_features():
    return [
        CloseDiff(),
        CloseToSma20(),
        CloseToLow(),
        HighToClose(),
    ]
