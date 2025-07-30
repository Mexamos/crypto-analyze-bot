import os
import logging

from dotenv import load_dotenv
from redis import Redis

from app.bot_controller import BotController
from app.cache.client import CacheClient
from app.database.client import DatabaseClient
from app.crypto.coindesk_client import CoindeskClient
from app.crypto.coingecko_client import CoingeckoClient
from app.crypto.coinmarketcap_client import CoinmarketcapClient
from app.crypto.cryptopanic_client import CryptopanicClient
from app.crypto.newsapi_client import NewsapiClient
from app.crypto.binance_client import BinanceClient
from app.crypto.santimentapi_client import SantimentApiClient
from app.crypto.santimentapi_model import ModelTrainingFacade, ModelPredictionFacade
from app.config import Config
from app.monitoring.sentry import SentryClient

load_dotenv()

TOKEN = os.getenv('BOT_TOKEN')
BOT_CHAT_IDS = os.getenv('BOT_CHAT_IDS', '').split(',')

COIN_MARKET_CAP_API_KEY = os.getenv('COIN_MARKET_CAP_API_KEY')
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY')
SANTIMENT_API_KEY = os.getenv('SANTIMENT_API_KEY')
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
CRYPTOPANIC_AUTH_TOKEN = os.getenv('CRYPTOPANIC_AUTH_TOKEN')

BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

SENTRY_DSN = os.getenv('SENTRY_DSN')

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = os.getenv('REDIS_PORT', 6379)
REDIS_DB = os.getenv('REDIS_DB', 0)
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')

TURN_ON_LOGS= os.getenv('TURN_ON_LOGS', 'False').lower() in ('true', '1', 'yes')

if TURN_ON_LOGS:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

# TODO Add for requests raises exceptions !!!!!!!!!!

# TODO порешать ошибки из сентри

# TODO написать Readme.md

# TODO написать тесты

# TODO добавить линтер(-ы)

def main():
    config = Config()
    db_client = DatabaseClient()
    # redis = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, password=REDIS_PASSWORD)
    cache_client = None # CacheClient(redis)
    sentry_client = SentryClient(SENTRY_DSN, config)

    coindesk_client = CoindeskClient('https://data-api.coindesk.com')
    coingecko_client = CoingeckoClient('https://api.coingecko.com', COINGECKO_API_KEY, config)
    cryptopanic_client = CryptopanicClient('https://cryptopanic.com', CRYPTOPANIC_AUTH_TOKEN)
    newsapi_client = NewsapiClient('https://newsapi.org', NEWS_API_KEY)
    santiment_api_client = SantimentApiClient(SANTIMENT_API_KEY)
    binance_client = BinanceClient(BINANCE_API_KEY, BINANCE_SECRET_KEY)
    coinmarketcap_client = CoinmarketcapClient(COIN_MARKET_CAP_API_KEY, config)

    model_training_facade = ModelTrainingFacade(api_client=santiment_api_client, config=config)
    model_prediction_facade = ModelPredictionFacade(api_client=santiment_api_client, config=config)

    telegram_controller = BotController(
        db_client,
        binance_client,
        coingecko_client,
        coindesk_client,
        cryptopanic_client,
        newsapi_client,
        santiment_api_client,
        coinmarketcap_client,
        model_training_facade,
        model_prediction_facade,
        cache_client,
        sentry_client,
        config,
        TOKEN,
        BOT_CHAT_IDS,
    )
    telegram_controller.get_all_coins()
    telegram_controller.init_bot()

    telegram_controller.run_bot()


if __name__ == "__main__":
    main()
