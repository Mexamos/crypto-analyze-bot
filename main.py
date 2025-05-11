import os

from dotenv import load_dotenv
from redis import Redis

from app.bot_controller import BotController
from app.database.client import DatabaseClient
from app.crypto.coindesk_client import CoindeskClient
from app.crypto.coingecko_client import CoingeckoClient
from app.crypto.cryptopanic_client import CryptopanicClient
from app.crypto.newsapi_client import NewsapiClient
from app.crypto.binance_client import BinanceClient
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

# TODO Add for requests raises exceptions !!!!!!!!!!

# TODO порешать ошибки из сентри

# TODO написать Readme.md

# TODO написать тесты

# TODO добавить линтер(-ы)

def main():
    config = Config()
    db_client = DatabaseClient()
    redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, password=REDIS_PASSWORD)

    coindesk_client = CoindeskClient('https://data-api.coindesk.com')
    coingecko_client = CoingeckoClient('https://api.coingecko.com', COINGECKO_API_KEY, config)
    cryptopanic_client = CryptopanicClient('https://cryptopanic.com', CRYPTOPANIC_AUTH_TOKEN)
    newsapi_client = NewsapiClient('https://newsapi.org', NEWS_API_KEY)
    binance_client = BinanceClient(BINANCE_API_KEY, BINANCE_SECRET_KEY)

    sentry_client = SentryClient(SENTRY_DSN, config)

    telegram_controller = BotController(
        db_client,
        binance_client,
        coingecko_client,
        coindesk_client,
        cryptopanic_client,
        newsapi_client,
        redis_client,
        sentry_client,
        config,
        TOKEN,
        BOT_CHAT_IDS,
    )
    telegram_controller.init_bot()
    telegram_controller.get_all_coins()
    telegram_controller.run_bot()


if __name__ == "__main__":
    main()


# import san
# import pandas as pd
# import datetime
# import joblib

# SANTIMENT_API_KEY = os.getenv('SANTIMENT_API_KEY')

# MODEL_PATH = "model.pkl"
# SCALER_PATH = "scaler.pkl"
# SLUGS = ["bitcoin", "ethereum"]
# TODAY = datetime.datetime.utcnow().date()
# FROM_DATE = (TODAY - datetime.timedelta(days=2)).strftime("%Y-%m-%d")
# TO_DATE = TODAY.strftime("%Y-%m-%d")
# INTERVAL = "1d"
# METRICS = [
#     "active_addresses_24h",
#     "active_addresses_24h_change_1d",
#     "active_addresses_24h_change_30d",
#     "30d_moving_avg_dev_activity_change_1d"
# ]

# san.ApiConfig.api_key = SANTIMENT_API_KEY

# def fetch_latest_data(slug):
#     features = {}
#     for metric in METRICS:
#         try:
#             df = san.get(metric, slug=slug, from_date=FROM_DATE, to_date=TO_DATE, interval=INTERVAL)
#             latest_value = df.iloc[-1]["value"]
#             features[metric] = latest_value
#         except Exception as e:
#             print(f"Error fetching {metric} for {slug}: {e}")
#             return None

#     return pd.Series(features)

# def predict(slug, model, scaler):
#     features = fetch_latest_data(slug)
#     if features is None:
#         return None

#     X_scaled = scaler.transform([features])
#     prediction = model.predict(X_scaled)[0]
#     return prediction

# def send_prediction():
#     model = joblib.load(MODEL_PATH)
#     scaler = joblib.load(SCALER_PATH)

#     lines = [f"🔮 *Prediction for {TODAY + datetime.timedelta(days=1):%B %d, %Y}*"]
#     for slug in SLUGS:
#         pred = predict(slug, model, scaler)
#         if pred:
#             arrow = {"UP": "🔼", "DOWN": "🔽", "STABLE": "⏸️"}[pred]
#             name = slug.capitalize()
#             lines.append(f"• *{name}* → {arrow} Likely to go *{pred}*")
#         else:
#             lines.append(f"• *{slug}* → ⚠️ Failed to predict")

#     message = "\n".join(lines)
#     print('message', message)

# # if __name__ == "__main__":
# #     send_prediction()


# projects_df = san.get("projects/all")
# print(projects_df[["name", "slug", "ticker"]].sort_values("name").to_string(index=False))


# now = datetime.datetime.utcnow()
# week_ago = now - timedelta(days=7)
# trends = san.get(
#     "emerging_trends",
#     from_date=week_ago.isoformat(),
#     to_date=now.isoformat(),
#     interval="1h",
#     size=10
# )

# print(trends["word"].unique())