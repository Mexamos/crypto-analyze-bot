import logging
import json
from asyncio import sleep as async_sleep
from datetime import datetime, timedelta
from typing import List, Dict
from functools import wraps
from time import sleep as sync_sleep

import pandas as pd
from pytz import timezone
from requests.exceptions import HTTPError
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from redis import Redis

from app.cache.client import CacheClient
from app.database.client import DatabaseClient
from app.crypto.coindesk_client import CoindeskClient
from app.crypto.coingecko_client import CoingeckoClient
from app.crypto.cryptopanic_client import CryptopanicClient
from app.crypto.binance_client import BinanceClient
from app.crypto.newsapi_client import NewsapiClient
from app.crypto.santimentapi_client import SantimentApiClient
from app.crypto.santimentapi_model import ModelTrainingFacade, ModelPredictionFacade
from app.config import Config
from app.monitoring.sentry import SentryClient


def retry_on_status_code(
    expected_status_code: int, max_retries: int = 3, backoff_factor: float = 1.0
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except HTTPError as exc:
                    status = getattr(exc.response, "status_code", None)
                    if status == expected_status_code:
                        wait_time = backoff_factor * (2 ** (attempt - 1))
                        sync_sleep(wait_time)
                    else:
                        raise

            return func(*args, **kwargs)
        return wrapper
    return decorator


class BotController:

    def __init__(
        self,
        db_client: DatabaseClient,
        binance_client: BinanceClient,
        coingecko_client: CoingeckoClient,
        coindesk_client: CoindeskClient,
        cryptopanic_client: CryptopanicClient,
        newsapi_client: NewsapiClient,
        santiment_api_client: SantimentApiClient,
        model_training_facade: ModelTrainingFacade,
        model_prediction_facade: ModelPredictionFacade,
        cache_client: CacheClient,
        sentry_client: SentryClient,
        config: Config,
        token: str,
        chat_ids: List[str],
    ) -> None:
        self.db_client = db_client
        self.binance_client = binance_client
        self.coingecko_client = coingecko_client
        self.coindesk_client = coindesk_client
        self.cryptopanic_client = cryptopanic_client
        self.newsapi_client = newsapi_client
        self.santiment_api_client = santiment_api_client
        self.model_training_facade = model_training_facade
        self.model_prediction_facade = model_prediction_facade

        self.cache_client: Redis = cache_client
        self.sentry_client = sentry_client
        self.config = config

        self.symbol_to_coingecko_id = {}
        self.timezone = timezone(self.config.timezone_name)
        self.date_format = self.config.common_date_format

        self.bot_token = token
        self.chat_ids = [int(chat_id) for chat_id in chat_ids]
        self.stopped = False

        self.sentiment_intensity_analyzer = SentimentIntensityAnalyzer()

    def init_bot(self):
        self.app = Application.builder().token(self.bot_token).build()

        self.app.add_handler(CommandHandler("health", self.health))
        # self.app.add_handler(CommandHandler("get_config", self.get_config))
        # self.app.add_handler(CommandHandler("change_config", self.change_config))
        self.app.add_handler(CommandHandler("get_last_trending_currencies", self.get_last_trending_currencies))
        self.app.add_handler(CommandHandler("scikit_learn_predict", self.scikit_learn_predict))
        self.app.add_handler(CommandHandler("stop", self.stop))

        job_queue = self.app.job_queue
        job_queue.run_repeating(
            self.process_trending_currencies, interval=self.config.analyze_task_interval, first=1
        )

    @retry_on_status_code(expected_status_code=429, backoff_factor=20)
    def get_all_coins(self):
        all_coins = self.coingecko_client._get_coins_list()
        self.symbol_to_coingecko_id = {
            coin["symbol"].upper(): coin["id"] 
            for coin in all_coins
        }

    def run_bot(self):
        # Run the bot until the user presses Ctrl-C
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)

    async def stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message.chat_id not in self.chat_ids:
            return

        jobs =  context.job_queue.jobs()
        for job in jobs:
            job.schedule_removal()

        self.app.stop_running()
        self.stopped = True
        await update.message.reply_text('Bot is stopping')

    async def health(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message.chat_id not in self.chat_ids:
            return

        jobs = context.job_queue.jobs()

        await update.message.reply_text(f'number_running_tasks={len(jobs)}')

    # async def get_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    #     if update.message.chat_id not in self.chat_ids:
    #         return

    #     parameter_list = '\n'.join([f'{name}={getattr(self.config, name)}' for name in self.config.parameter_list])
    #     await update.message.reply_text(parameter_list)

    # async def change_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    #     if update.message.chat_id not in self.chat_ids:
    #         return

    #     if len(context.args) < 2:
    #         await update.message.reply_text('Need config parameter name and new value as comand arguments!')
    #         return

    #     try:
    #         self.config.change_value(context.args[0], context.args[1])
    #     except Exception as ex:
    #         self.sentry_client.capture_exception(ex)

    def _get_trending_coins(self):
        response = self.coingecko_client._get_trending_coins()
        return [entry['item'] for entry in response.get('coins', [])]

    async def _get_coin_sentiment(
        self, coin_id: str, max_retries: int = 3, backoff_factor: float = 1.0
    ) -> float:
        for attempt in range(1, max_retries + 1):
            if self.stopped:
                break

            try:
                response = self.coingecko_client._get_coin_by_id(coin_id)
                up_pct = response.get('sentiment_votes_up_percentage', 0) or 0
                down_pct = response.get('sentiment_votes_down_percentage', 0) or 0

                total = up_pct + down_pct
                return (up_pct / total) if total else 0.5
            except HTTPError as exc:
                status = getattr(exc.response, 'status_code', None)
                if status == 429:
                    retry_after = exc.response.headers.get('Retry-After')
                    wait_time = (
                        int(retry_after) if retry_after and retry_after.isdigit() else 60
                    ) + backoff_factor * (attempt - 1)
                    logging.warning(f'[{coin_id}] 429, sleeping {wait_time}s (attempt {attempt})')
                    await async_sleep(wait_time)
                    continue

                self.sentry_client.capture_exception(Exception(f'[{coin_id}] HTTP {status}: {exc}, returning 0.5'))
                return 0.5

        self.sentry_client.capture_exception(Exception(f'[{coin_id}] Failed {max_retries} attempts, returning 0.5'))
        return 0.5

    def _get_coindesk_articles(self, lang: str = "EN", limit: int = 10) -> list:
        response = self.coindesk_client._get_news_article_list(
            lang=lang, limit=limit
        )
        return response.get("Data", [])

    def _get_cryptopanic_posts(self, limit: int = 50) -> list:
        response = self.cryptopanic_client._get_free_posts(limit)
        return response.get("results", [])

    def _get_santiment_api_trend_words(self):
        to_date = datetime.now()
        from_date = to_date - timedelta(days=1)

        trend_words = self.santiment_api_client.emerging_trends(
            from_date=from_date.isoformat(),
            to_date=to_date.isoformat(),
        )
        valid_trend_words = []
        for trend_word in trend_words:
            if trend_word.upper() in self.symbol_to_coingecko_id.keys():
                valid_trend_words.append(trend_word.upper())

        return valid_trend_words

    def _get_news_sentiment(self, query: str) -> float:
        try:
            response = self.newsapi_client._get_everything(query)
            articles = response.get("articles", [])
        except HTTPError as exc:
            exception_body = exc.response.json()
            if exception_body.get('code') != 'rateLimited':
                self.sentry_client.capture_exception(
                    Exception(f'NewsAPI failed for {query}: {exception_body}')
                )

            return 0.5

        scores = [
            self.sentiment_intensity_analyzer.polarity_scores(
                article.get("title","") + " " + (article.get("description") or "")
            )["compound"]
            for article in articles
        ]
        return (sum(scores) / len(scores)) if scores else 0.5

    async def _filter_currencies_by_binance(self, currencies: List[dict]) -> List[dict]:
        available_currencies = []
        for currency in currencies:
            try:
                exchange_info = self.binance_client.exchange_info(currency['exchange_symbol'])
                if exchange_info:
                    available_currencies.append(currency)
            except HTTPError as exc:
                exception_body = exc.response.json()
                if not 'Invalid symbol' in exception_body.get('msg'):
                    self.sentry_client.capture_exception(
                        Exception(f'Get Binance exchange_info failed for {currency["exchange_symbol"]}: {exception_body}')
                    )
                continue

        return available_currencies

    async def _filter_conversion_currency(self, currencies: List[dict]) -> List[dict]:
        available_currencies = []
        for currency in currencies.values():
            if currency['symbol'] != self.config.currency_conversion:
                available_currencies.append(currency)

        return available_currencies

    async def _forming_currencies_list(
        self, trending_coins, coindesk_articles, cryptopanic_posts, trend_words
    ) -> Dict[str, str]:
        coin_info: dict[str, dict[str, str]] = {}
        seen = set()
        for coin in trending_coins:
            cid = coin['id']
            symbol = coin['symbol'].upper()
            if symbol in seen:
                continue

            coin_info[cid] = {
                'symbol': symbol,
                'exchange_symbol': symbol + self.config.currency_conversion,
                'name': coin['name']
            }
            seen.add(symbol)

        for article in coindesk_articles:
            for asset in article.get('assets', []):
                slug = asset.get('slug', '').lower()
                symbol = slug.upper()
                if not slug or slug in coin_info or symbol in seen:
                    continue

                coin_info[slug] = {
                    'symbol': symbol,
                    'exchange_symbol': symbol + self.config.currency_conversion,
                    'name': asset.get('name', slug)
                }
                seen.add(symbol)

        for post in cryptopanic_posts:
            for cur in post.get('currencies', []):
                slug = cur.get('code', '').lower()
                symbol = slug.upper()
                if not slug or slug in coin_info or symbol in seen:
                    continue

                coin_info[slug] = {
                    'symbol': symbol,
                    'exchange_symbol': symbol + self.config.currency_conversion,
                    'name': symbol
                }
                seen.add(symbol)

        for trend_word in trend_words:
            if trend_word in seen:
                continue

            coin_info[trend_word] = {
                'symbol': trend_word,
                'exchange_symbol': trend_word + self.config.currency_conversion,
                'name': trend_word
            }
            seen.add(trend_word)

        coin_info = await self._filter_conversion_currency(coin_info)
        return await self._filter_currencies_by_binance(coin_info)

    async def _get_metrics_per_currency(
        self, coin_info, cryptopanic_posts
    ):
        rows: List[dict[str, float | str]] = []
        for info in coin_info:
            if self.stopped:
                break

            symbol = info['symbol']
            name = info['name']
            coingecko_id = self.symbol_to_coingecko_id.get(symbol)

            comm_score = await self._get_coin_sentiment(coingecko_id)
            await async_sleep(1)

            news_score = self._get_news_sentiment(name)

            cp_posts = [
                p for p in cryptopanic_posts
                if any(c.get('code', '').lower() == symbol.lower() for c in p.get('currencies', []))
            ]
            cp_score = sum(
                p.get('votes', {}).get('positive', 0) - p.get('votes', {}).get('negative', 0)
                for p in cp_posts
            )

            rows.append({
                "symbol": symbol,
                "comm_score": comm_score,
                "news_score": news_score,
                "cryptopanic_score": cp_score,
            })

        return rows

    async def _calculate_signals(self, rows) -> pd.DataFrame:
        # Build DataFrame and normalize
        df = pd.DataFrame(rows)
        for col in ["comm_score", "news_score", "cryptopanic_score"]:
            mn, mx = df[col].min(), df[col].max()
            df[f"{col}_n"] = (df[col] - mn) / (mx - mn) if mx > mn else 0.5

        # Compute composite score (renormalized weights) and signal
        # Original weights: 0.5, 0.2, 0.2 -> sum=0.9; normalized -> 0.556, 0.222, 0.222
        df["composite"] = (
            0.556 * df["comm_score_n"] +
            0.222 * df["news_score_n"] +
            0.222 * df["cryptopanic_score_n"]
        )
        df["signal"] = df["composite"].apply(
            lambda x: "BUY" if x > 0.7 else ("SELL" if x < 0.3 else "HOLD")
        )
        return df

    async def _filter_records(self, df: pd.DataFrame, cached_records):
        last_records = json.loads(cached_records)
        last_records_symbol = last_records['symbol']
        last_records_signal = last_records['signal']
        prev_pairs = {(last_records_symbol[key], last_records_signal[key]) for key in last_records_symbol}

        return df[df.apply(lambda row: (row['symbol'], row['signal']) not in prev_pairs, axis=1)]

    async def _generate_massage(self, records):
        message_lines = [
            f"{symbol}" + ((12 - len(symbol)) * ' ') + f"{composite:.3f} ({signal})"
            for symbol, composite, signal in zip(
                records['symbol'].values(), records['composite'].values(), records['signal'].values()
            )
        ]
        if not message_lines:
            return

        return '```\n' + "\n".join(message_lines) + '```'

    async def process_trending_currencies(self, context: ContextTypes.DEFAULT_TYPE):
        try:
            trending_coins = self._get_trending_coins()
            coindesk_articles = self._get_coindesk_articles(limit=10)
            cryptopanic_posts = self._get_cryptopanic_posts(limit=50)
            trend_words = self._get_santiment_api_trend_words()

            coin_info = await self._forming_currencies_list(
                trending_coins, coindesk_articles, cryptopanic_posts, trend_words
            )

            rows = await self._get_metrics_per_currency(coin_info, cryptopanic_posts)

            df = await self._calculate_signals(rows)

            df = df.sort_values("composite", ascending=False)[["symbol", "composite", "signal"]]
            new_records = df.to_dict()

            cached_records = self.cache_client.get('last_trending_currencies')
            if cached_records:
                df = await self._filter_records(df, cached_records)

            self.cache_client.set('last_trending_currencies', json.dumps(new_records))

            message = await self._generate_massage(df.to_dict())
            if cached_records and message:
                for chat_id in self.chat_ids:
                    await context.bot.send_message(chat_id, message, parse_mode=ParseMode.MARKDOWN_V2)

        except Exception as ex:
            self.sentry_client.capture_exception(ex)

    async def get_last_trending_currencies(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message.chat_id not in self.chat_ids:
            return

        cached_records = self.cache_client.get('last_trending_currencies')
        if cached_records:
            last_records = json.loads(cached_records)

            message = await self._generate_massage(last_records)
            if not message:
                return

            await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN_V2)

    async def scikit_learn_predict(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message.chat_id not in self.chat_ids:
            return

        if len(context.args) < 1:
            await update.message.reply_text('Need blockchain currency name as comand argument!')
            return

        try:
            currency_name = context.args[0]
            start_training_date = (
                context.args[1] if len(context.args) > 1
                else (datetime.now(self.timezone) - timedelta(days=365)).strftime(self.date_format)
            )
            end_training_date = datetime.now(self.timezone).strftime(self.date_format)

            self.santiment_api_client.validate_slugs([currency_name])
            self.model_training_facade.train_and_save_model(
                [currency_name], start_training_date, end_training_date
            )
            message = self.model_prediction_facade.predict([currency_name])

            await update.message.reply_text(message)

        except Exception as ex:
            self.sentry_client.capture_exception(ex)
            await update.message.reply_text(f'Error in scikit_learn_predict: {ex}')
