from model.features.close_price_prct_diff import CloseDiff
from model.features.close_to_low import LowToCloseRatio
from model.features.high_to_close import HighToClose
from model.features.sma10 import Sma10


def get_features():
    return [
        CloseDiff(),
        Sma10(),
        LowToCloseRatio(),
        HighToClose(),
    ]
