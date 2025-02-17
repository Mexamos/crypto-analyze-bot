import os

from dotenv import load_dotenv
from redis import Redis

from app.bot_controller import BotController
from app.crypto.binance_client import BinanceClient
from app.config import Config
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

# TODO Записать все хочты и порты в конфиг и брать оттуда

# TODO Add for requests raises exceptions !!!!!!!!!!

# TODO добавить сентри

# TODO написать Readme.md

# TODO написать тесты

# TODO добавить линтер(-ы)

def main():
    config = Config()

    binance_cleint = BinanceClient(BINANCE_API_KEY, BINANCE_SECRET_KEY)
    redis_client = Redis(host='redis', port=6379, db=0)

    # sentry_client = SentryClient(SENTRY_DSN, config)

    bot_controller = BotController(
        config, binance_cleint, redis_client
    )
    bot_controller.run_bot()



if __name__ == "__main__":
    main()
