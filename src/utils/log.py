import logging
from logging import DEBUG

from config import LOG_PATH


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    file_name_format = " (%(filename)s:%(lineno)d)"
    format = "[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + file_name_format + reset,
        logging.ERROR: red + format + file_name_format + reset,
        logging.CRITICAL: bold_red + format + file_name_format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(DEBUG)

    file_handler = logging.FileHandler(LOG_PATH + '/app.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s]: %(message)s (%(filename)s:%(lineno)d)"))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(CustomFormatter())

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
