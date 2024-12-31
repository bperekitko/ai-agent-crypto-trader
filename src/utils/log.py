from datetime import datetime


class Logger:
    __NO_COLOR = 0
    __YELLOW = 33
    __RED = 31
    def __init__(self, name):
        self.name = name

    def info(self, message):
        self.__log(message, 'INFO', self.__NO_COLOR)

    def warn(self, message):
        self.__log(message, 'WARN', self.__YELLOW)

    def error(self, message):
        self.__log(message, 'ERROR', self.__RED)

    def __log(self, message, level, color):
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f'\033[{color}m [{level}] [{formatted_time}] [{self.name}]: {message}\033[0m')
