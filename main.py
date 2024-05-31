import os

from dotenv import load_dotenv

from app.telegram_controller import TelegramController
from app.database.client import DatabaseClient
from app.coinmarketcap_client import CoinmarketcapClient
from app.binance_client import BinanceClient
from app.chart import ChartController
from app.config import Config
from app.google_sheets_client import GoogleSheetsClient
from app.sentry import SentryClient

load_dotenv()

TOKEN = os.getenv('BOT_TOKEN')
BOT_CHAT_ID = os.getenv('BOT_CHAT_ID')

COIN_MARKET_CAP_API_KEY = os.getenv('COIN_MARKET_CAP_API_KEY')

BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

CREDENTIALS_FILE_PATH = os.getenv('CREDENTIALS_FILE_PATH')
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')

SENTRY_DSN = os.getenv('SENTRY_DSN')

# TODO перегруппировать весь код -> проверить на проде, с уменьшенным конфигом

# TODO Изменить стратегию
# existed_list, если пустой, добавить туда все
# если не пустой, то смотрим каких валют нет в этом списке
# купить отсутствующие валюты, если возможно
# если валюты нет в списке existed_list, но есть в БД, то пометить что она вышла из трендов и нужно ее продать любой ценой
# если валюта встречается в списке existed_list, и ее новая цена больше на 3% или 100$, то продать

# TODO добавить команды в телеграм бот


# TODO подумать на счет валют которые долго висят на балансе и не продаются

# TODO написать Readme.md

def main():
    config = Config()
    db_client = DatabaseClient()

    chart_controller = ChartController(db_client, config)
    google_sheets_client = GoogleSheetsClient(CREDENTIALS_FILE_PATH, SPREADSHEET_ID)

    cmc_client = CoinmarketcapClient(COIN_MARKET_CAP_API_KEY, config)
    binance_cleint = BinanceClient(BINANCE_API_KEY, BINANCE_SECRET_KEY)

    sentry_client = SentryClient(SENTRY_DSN, config)

    telegram_controller = TelegramController(
        db_client, cmc_client, binance_cleint,
        chart_controller, google_sheets_client,
        sentry_client, config, TOKEN, BOT_CHAT_ID
    )
    telegram_controller.restore_unsold_currencies()
    telegram_controller.run_bot()


if __name__ == "__main__":
    main()
