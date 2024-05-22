import os

from dotenv import load_dotenv

from telegram_controller import TelegramController
from database.client import DatabaseClient
from coinmarketcap_client import CoinmarketcapClient
from binance_client import BinanceClient
from chart import ChartController
from config import Config
from google_sheets_client import GoogleSheetsClient

load_dotenv()

TOKEN = os.getenv('BOT_TOKEN')
BOT_CHAT_ID = os.getenv('BOT_CHAT_ID')

COIN_MARKET_CAP_API_KEY = os.getenv('COIN_MARKET_CAP_API_KEY')

BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

CREDENTIALS_FILE_PATH = os.getenv('CREDENTIALS_FILE_PATH')
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')


# TODO добавить команду health
# проверить что таска работает
# получить последнюю запись в таблицу CurrencyPrice
# Сделать запись в гугл таблицу и вернуть ссылку на эту тестовую таблицу
# Добавить в класс CoinmarketcapClient параметр дата последнего запроса, возвращать эту дату

# TODO Считывать данные из Unsold и добавлять в таблицу CurrencyPrice перед запуском

# /Users/mexamos/code/python-lang/crypto-analyze-bot/venv/lib/python3.9/site-packages/pandas/plotting/_matplotlib/core.py:580: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
#   fig = self.plt.figure(figsize=self.figsize)



# TODO остановка торговли - постепенная остановка и продажа всего

# TODO подумать на счет валют которые долго висят на балансе и не продаются

# TODO Добавить команду завершения торговли
# Прекращает покупку, при продаже последней валюты, останавливает бота

# TODO написать Readme.md

def main():
    config = Config()
    db_client = DatabaseClient()

    chart_controller = ChartController(db_client, config)
    google_sheets_client = GoogleSheetsClient(CREDENTIALS_FILE_PATH, SPREADSHEET_ID)

    cmc_client = CoinmarketcapClient(COIN_MARKET_CAP_API_KEY)
    binance_cleint = BinanceClient(BINANCE_API_KEY, BINANCE_SECRET_KEY)

    telegram_controller = TelegramController(
        db_client, cmc_client, binance_cleint, chart_controller, google_sheets_client, config, TOKEN, BOT_CHAT_ID
    )
    telegram_controller.run_bot()


if __name__ == "__main__":
    main()
