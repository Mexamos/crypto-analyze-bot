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

# TODO Добавить к списку текущих валют дату покупки

# TODO Change timezone

# TODO Возможно добавить изменение конфига на лету

# TODO Добавить отлавливание остановки бота и слать сообщение, что бот был остановлен
# + слать отчет по прибыли и по не проданным валютам

# TODO Добавить команду завершения торговли
# Прекращает покупку, при продаже последней валюты, останавливает бота

# TODO Проверить и вероятно добавить асинхронность к запросам request

# TODO добавить команду health

# TODO написать Readme.md

# TODO запустить на удаленном сервере

# TODO Filter from list of currencies USDT currency

# TODO подумать на счет валют которые долго висят на балансе и не продаются

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
