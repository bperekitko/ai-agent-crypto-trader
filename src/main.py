from binance.um_futures import UMFutures
from config import BINANCE_FUTURES_API_KEY, BINANCE_FUTURES_SECRET_KEY
import time
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient

from utils.log import get_logger


# Get account information
def message_handler(_, message):
    print(message)

def main():
    # cm_futures_client = UMFutures(base_url="https://testnet.binancefuture.com/")
    # um_futures_client = UMFutures(key="448df9304ad592cf9a0a7cd03483c6eb0baf7a42b18c5dc4b7028ad1e9198a26", secret="e515956c975451b9882a2abf63d6efc8346ddefee2da4db57cf568a636c8416f",base_url="https://testnet.binancefuture.com")
    # print(um_futures_client.time())
    #
    # account = um_futures_client.account()
    # for asset in account['assets']:
    #     print(asset['asset'], asset['walletBalance'])
    # #
    # my_client = UMFuturesWebsocketClient(on_message=message_handler, stream_url="wss://stream.binancefuture.com")
    # #
    #
    # try:
    #     response = um_futures_client.get_all_orders(symbol="BTCUSDT", recvWindow=2000)
    #     logging.info(response)
    # except ClientError as error:
    #     logging.error(
    #         "Found error. status: {}, error code: {}, error message: {}".format(
    #             error.status_code, error.error_code, error.error_message
    #         )
    #     )
    #
    # my_client.kline(
    #     symbol="btcusdt",
    #     interval="1h"
    # )
    #
    # time.sleep(125)
    # my_client.send_message_to_server()
    #
    # print("closing ws connection")
    # my_client.stop()
    logger = get_logger(__name__)
    logger.info("TESTING")
    logger.warning("THIS IS A WARNIGN")
    logger.error("THIS IS AN ERROR")
    logger.critical("THIS IS A CRITITCAL")


if __name__ == "__main__":
    main()
