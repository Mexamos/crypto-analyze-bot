import os
from datetime import datetime

from dotenv import load_dotenv
from redis import Redis

from app.bot_controller import BotController
from app.crypto.binance_client import BinanceClient
from app.config import Config
from app.monitoring.sentry import SentryClient

load_dotenv()

BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')

SENTRY_DSN = os.getenv('SENTRY_DSN')

# TODO Записать все хосты и порты в конфиг и брать оттуда

# TODO Add for requests raises exceptions !!!!!!!!!!

# TODO добавить сентри

# TODO написать Readme.md

# TODO написать тесты

# TODO добавить линтер(-ы)

def main():
    config = Config()

    binance_cleint = BinanceClient(BINANCE_API_KEY, BINANCE_SECRET_KEY)
    redis_client = Redis(host=REDIS_HOST, port=6379, db=0, password=REDIS_PASSWORD)

    # sentry_client = SentryClient(SENTRY_DSN, config)

    # bot_controller = BotController(
    #     config, binance_cleint, redis_client
    # )
    # bot_controller.request_static_data()
    # bot_controller.run_bot()

# if __name__ == "__main__":
#     main()



import requests
from datetime import datetime, timedelta, timezone

def get_crypto_news():
    url = "https://cryptopanic.com/api/free/v1/posts/"
    params = {
        "auth_token": "8996a617d6e3e48c9edeafa8e54810426352e6b5",
        # "currencies": "BTC",
        "filter": "hot",
        "public": "true"
    }
    r = requests.get(url, params=params)
    return r.json().get("results", [])

def analyze_post(post):
    print('post', post)
    votes = post.get("votes", {})
    pos = votes.get("positive", 0)
    neg = votes.get("negative", 0)
    total = pos + neg + 1
    score = (pos - neg) / total

    published = datetime.fromisoformat(post["published_at"].replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    is_recent = now - published < timedelta(days=1)

    return score if is_recent else 0

# hot_posts = get_crypto_news()
# total_score = 0
# for post in hot_posts:
#     total_score += analyze_post(post)

# total_score = total_score / len(hot_posts)
# print('total_score', total_score)
# print('The result should be between -1 and 1')


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








import os
import requests
import time
import logging
from requests.exceptions import HTTPError
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Конфиг ---
HEADERS = {
    "accept": "application/json",
    "x-cg-api-key": "CG-TpiMzNseLaFZQmyL243SPSZ7"
}
session = requests.Session()
session.headers.update(HEADERS)

TRENDING_URL   = "https://api.coingecko.com/api/v3/search/trending"
NEWS_API_KEY   = '371db309258d48a498e90303233ef691'
analyzer       = SentimentIntensityAnalyzer()

def fetch_trending_coins():
    """
    Возвращает список трендовых монет (по умолчанию top 15).
    Каждая запись — dict с ключами: id, name, symbol, market_cap_rank и т.п.
    """
    r = requests.get(TRENDING_URL, headers=HEADERS)
    r.raise_for_status()
    items = r.json().get("coins", [])
    return [entry["item"] for entry in items]

def fetch_coin_sentiment(
    coin_id: str,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
):
    """
    Берёт sentiment_votes_up/down (%) из community_data
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    params = {
        "localization":      "false",
        "tickers":           "false",
        "market_data":       "false",
        "community_data":    "false",
        "developer_data":    "false",
        "sparkline":         "false",
    }
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, params=params)
            r.raise_for_status()
            response = r.json()

            up_pct   = response.get("sentiment_votes_up_percentage",   0)
            down_pct = response.get("sentiment_votes_down_percentage", 0)
            # Приводим к [0–1]
            total = up_pct + down_pct
            return (up_pct/total) if total else 0.5

        except HTTPError as e:
            status = getattr(e.response, "status_code", None)
            # Если rate limit — ждём и пробуем снова
            if status == 429:
                ra = e.response.headers.get("Retry-After")
                wait = (
                    (int(ra) if ra and ra.isdigit() else 60) + backoff_factor * (attempt - 1)
                )
                logging.warning(
                    f"[{coin_id}] 429 Too Many Requests — спим {wait}s "
                    f"(попытка {attempt}/{max_retries})"
                )
                time.sleep(wait)
                continue
            # Для других ошибок — логируем и возвращаем нейтральный
            logging.error(f"[{coin_id}] HTTP {status}: {e}, ставим 0.5")
            return 0.5

def get_news_sentiment(query: str):
    """
    Достаёт заголовки/описания из NewsAPI и считает compound‑оценку VADER
    """
    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "pageSize": 10, "apiKey": NEWS_API_KEY}
    art = requests.get(url, params=params).json().get("articles", [])
    scores = [
        analyzer.polarity_scores(a["title"] + " " + (a.get("description") or ""))["compound"]
        for a in art
    ]
    return (sum(scores) / len(scores)) if scores else 0

# --- Основная логика ---
# 1) Берём трендовые монеты
trending = fetch_trending_coins()  # список до 15 items

print('trending', len(trending))
# 2) Сбор данных
rows = []
for coin in trending:
    cid    = coin["id"]       # строковый coin_id, например "bitcoin"
    sym    = coin["symbol"]   # например "btc"
    # 2.1) Комьюнити‑сентимент
    comm_score = fetch_coin_sentiment(cid)
    # 2.2) Новостной сентимент (опционально)
    news_score = get_news_sentiment(coin["name"])
    rows.append({
        "symbol":     sym.upper(),
        "comm_score": comm_score,
        "news_score": news_score
    })

print('rows', rows)
print('rows', len(rows))
df = pd.DataFrame(rows)

# 3) Нормализация
for col in ("comm_score", "news_score"):
    mn, mx = df[col].min(), df[col].max()
    print('mn', mn)
    print('mx', mx)
    df[col + "_n"] = (df[col] - mn) / (mx - mn) if mx > mn else 0.5

# 4) Composite Score и сигнал
df["composite"] = 0.7 * df["comm_score_n"] + 0.3 * df["news_score_n"]
df["signal"] = df["composite"].apply(
    lambda x: "BUY"  if x > 0.7 else ("SELL" if x < 0.3 else "HOLD")
)

# 5) Вывод
print(df.sort_values("composite", ascending=False)[["symbol","composite","signal"]])
