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

# TODO Add for requests raises exceptions !!!!!!!!!!

# TODO urls
# /v2/cryptocurrency/quotes/historical
# /v1/cryptocurrency/listings/historical

# TODO порешать ошибки из сентри

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
    telegram_controller.init_bot()
    # telegram_controller.restore_unsold_currencies()
    telegram_controller.run_bot()


# if __name__ == "__main__":
#     main()




from binance.client import Client
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

binance_cleint = BinanceClient(BINANCE_API_KEY, BINANCE_SECRET_KEY)


def place_market_order():

    AMOUNT_USDT = 10  # Amount of USDT to spend (start with small amounts!)

    # Check balance first
    balance = binance_cleint.get_account_balance('SOL')
    print('balance', balance)
    
    if not balance or balance < AMOUNT_USDT:
        print(f"Insufficient USDT balance. Available: {balance} USDT")
        return

    # binance_cleint.make_order('SOLUSDT', AMOUNT_USDT)

    # binance_cleint.make_convertation(
    #     from_asset='USDT', to_asset='TON', from_amount=10
    # )


# place_market_order()