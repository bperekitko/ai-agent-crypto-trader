import os

__CURRENT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))


def trading_dir_path(filename):
    return os.path.join(__CURRENT_DIR_PATH, filename)