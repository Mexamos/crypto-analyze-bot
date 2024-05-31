import os

from dotenv import load_dotenv

from app.bot_controller import BotController
from app.database.client import DatabaseClient
from app.crypto.coinmarketcap_client import CoinmarketcapClient
from app.crypto.binance_client import BinanceClient
from app.analytics.chart import ChartController
from app.config import Config
from app.analytics.google_sheets_client import GoogleSheetsClient
from app.monitoring.sentry import SentryClient

load_dotenv()

TOKEN = os.getenv('BOT_TOKEN')
BOT_CHAT_ID = os.getenv('BOT_CHAT_ID')

COIN_MARKET_CAP_API_KEY = os.getenv('COIN_MARKET_CAP_API_KEY')

BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

CREDENTIALS_FILE_PATH = os.getenv('CREDENTIALS_FILE_PATH')
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')

SENTRY_DSN = os.getenv('SENTRY_DSN')

# TODO Изменить стратегию
# existed_list, если пустой, добавить туда все
# если не пустой, то смотрим каких валют нет в этом списке
# купить отсутствующие валюты, если возможно
# если валюты нет в списке existed_list, но есть в БД, то пометить что она вышла из трендов и нужно ее продать любой ценой
# если валюта встречается в списке existed_list, и ее новая цена больше на 3% или 100$, то продать


# TODO подумать на счет валют которые долго висят на балансе и не продаются

# TODO написать Readme.md

# TODO написать тесты

# TODO добавить линтер(-ы)

def main():
    config = Config()
    db_client = DatabaseClient()

    chart_controller = ChartController(db_client, config)
    google_sheets_client = GoogleSheetsClient(CREDENTIALS_FILE_PATH, SPREADSHEET_ID)

    cmc_client = CoinmarketcapClient(COIN_MARKET_CAP_API_KEY, config)
    binance_cleint = BinanceClient(BINANCE_API_KEY, BINANCE_SECRET_KEY)

    sentry_client = SentryClient(SENTRY_DSN, config)

    telegram_controller = BotController(
        db_client, cmc_client, binance_cleint,
        chart_controller, google_sheets_client,
        sentry_client, config, TOKEN, BOT_CHAT_ID
    )
    telegram_controller.restore_unsold_currencies()
    telegram_controller.run_bot()


if __name__ == "__main__":
    main()
