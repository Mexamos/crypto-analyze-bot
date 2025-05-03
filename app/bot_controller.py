import logging
import math
from asyncio import sleep
from datetime import datetime
from decimal import Decimal
from typing import List, Set, Dict

import pandas as pd
import requests
from requests.exceptions import HTTPError
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes
from pytz import timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.database.client import DatabaseClient
from app.database.models import CurrencyPrice
from app.crypto.coindesk_client import CoindeskClient
from app.crypto.coingecko_client import CoingeckoClient
from app.crypto.coinmarketcap_client import CoinmarketcapClient, CmcException
from app.crypto.cryptopanic_client import CryptopanicClient
from app.crypto.binance_client import BinanceClient
from app.crypto.newsapi_client import NewsapiClient
from app.analytics.chart import ChartController, DataForChartNotFound, DataForIncomesTableNotFound
from app.config import Config
from app.analytics.google_sheets_client import GoogleSheetsClient, GoogleSheetAppendIncomeFailed
from app.utils import scientific_notation_to_usual_format
from app.monitoring.sentry import SentryClient


class BotController:

    def __init__(
        self,
        db_client: DatabaseClient,
        binance_client: BinanceClient,
        coingecko_client: CoingeckoClient,
        coindesk_client: CoindeskClient,
        cryptopanic_client: CryptopanicClient,
        newsapi_client: NewsapiClient,
        config: Config,
        token: str,
        chat_id: str,
    ) -> None:
        self.db_client = db_client
        self.binance_client = binance_client
        self.coingecko_client = coingecko_client
        self.coindesk_client = coindesk_client
        self.cryptopanic_client = cryptopanic_client
        self.newsapi_client = newsapi_client
        self.config = config

        self.symbol_to_coingecko_id = {}

        self.bot_token = token
        self.chat_id = int(chat_id)

        self.known_currencies = set()

        self.out_of_trend_currencies = set()

        self.timezone = timezone(self.config.timezone_name)
        self.launch_datetime = datetime.now(self.timezone)

        self.stop_buying_flag = False

        self.sentiment_intensity_analyzer = SentimentIntensityAnalyzer()

    def init_bot(self):
        self.app = Application.builder().token(self.bot_token).build()

        # self.app.add_handler(CommandHandler("health", self.health))
        # self.app.add_handler(CommandHandler("get_config", self.get_config))
        # self.app.add_handler(CommandHandler("change_config", self.change_config))
        # self.app.add_handler(CommandHandler("stop_buying", self.stop_trading))
        # self.app.add_handler(CommandHandler("start_buying", self.start_trading))
        # self.app.add_handler(CommandHandler("stop", self.stop))

        job_queue = self.app.job_queue
        job_queue.run_repeating(
            self.process_trending_currencies, interval=self.config.process_task_interval, first=1
        )

    def get_all_coins(self):
        all_coins = self.coingecko_client._get_coins_list()
        self.symbol_to_coingecko_id = {
            coin["symbol"].upper(): coin["id"] 
            for coin in all_coins
        }

    def run_bot(self):
        # Run the bot until the user presses Ctrl-C
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)

    # async def stop_trading(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    #     if update.message.chat_id != self.chat_id:
    #         return

    #     self.stop_buying_flag = True

    # async def start_trading(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    #     if update.message.chat_id != self.chat_id:
    #         return

    #     self.stop_buying_flag = False

    # async def stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    #     if update.message.chat_id != self.chat_id:
    #         return

    #     jobs =  context.job_queue.jobs()
    #     for job in jobs:
    #         job.schedule_removal()

    #     await self._sell_or_record_to_table_rest_currencies(context)

    #     await self._generate_incomes_report()

    #     self.app.stop_running()
    #     await update.message.reply_text('Bot stopped')

    # async def health(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    #     if update.message.chat_id != self.chat_id:
    #         return

    #     jobs = context.job_queue.jobs()

    #     currency_prices_count = self.db_client.count_all_currency_price()

    #     self.google_sheets_client.append_to_test_connection(datetime.now(self.timezone))
    #     google_sheet_link = f'https://docs.google.com/spreadsheets/d/{self.google_sheets_client.spreadsheet_id}/'

    #     coin_market_cap_latest_request_datetime = (
    #         self.cmc_client.latest_request_datetime.strftime("%d.%m.%Y %H:%M:%S")
    #         if self.cmc_client.latest_request_datetime else None
    #     )

    #     await update.message.reply_text(f'number_running_tasks={len(jobs)}')
    #     await update.message.reply_text(f'currency_prices_count={currency_prices_count}')
    #     await update.message.reply_text(f'google_sheet_link={google_sheet_link}')
    #     await update.message.reply_text(f'coin_market_cap_latest_request_datetime={coin_market_cap_latest_request_datetime}')

    # async def get_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    #     if update.message.chat_id != self.chat_id:
    #         return

    #     parameter_list = '\n'.join([f'{name}={getattr(self.config, name)}' for name in self.config.parameter_list])
    #     await update.message.reply_text(parameter_list)

    # async def change_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    #     if update.message.chat_id != self.chat_id:
    #         return

    #     if len(context.args) < 2:
    #         await update.message.reply_text('Need config parameter name and new value as comand arguments!')
    #         return

    #     try:
    #         self.config.change_value(context.args[0], context.args[1])
    #     except BaseException as ex:
    #         self.sentry_client.capture_exception(ex)

    def _get_trending_coins(self):
        response = self.coingecko_client._get_trending_coins()
        return [entry['item'] for entry in response.get('coins', [])]

    async def _get_coin_sentiment(
        self, coin_id: str, max_retries: int = 3, backoff_factor: float = 1.0
    ) -> float:
        # for attempt in range(1, max_retries + 1):
        #     try:
        #         response = self.coingecko_client._get_coin_by_id(coin_id)
        #         up_pct = response.get('sentiment_votes_up_percentage', 0) or 0
        #         down_pct = response.get('sentiment_votes_down_percentage', 0) or 0

        #         total = up_pct + down_pct
        #         return (up_pct / total) if total else 0.5
        #     except HTTPError as exc:
        #         status = getattr(exc.response, 'status_code', None)
        #         if status == 429:
        #             retry_after = exc.response.headers.get('Retry-After')
        #             wait_time = (
        #                 int(retry_after) if retry_after and retry_after.isdigit() else 60
        #             ) + backoff_factor * (attempt - 1)
        #             logging.warning(f'[{coin_id}] 429, sleeping {wait_time}s (attempt {attempt})')
        #             await sleep(wait_time)
        #             continue

        #         logging.error(f'[{coin_id}] HTTP {status}: {exc}, returning 0.5')
        #         return 0.5

        # logging.error(f'[{coin_id}] Failed {max_retries} attempts, returning 0.5')
        return 0.5

    def _get_coindesk_articles(self, lang: str = "EN", limit: int = 10) -> list:
        response = self.coindesk_client._get_news_article_list(
            lang=lang, limit=limit
        )
        return response.get("Data", [])

    def _get_cryptopanic_posts(self, limit: int = 50) -> list:
        response = self.cryptopanic_client._get_free_posts(limit)
        return response.get("results", [])

    def get_news_sentiment(self, query: str) -> float:
        try:
            response = self.newsapi_client._get_everything(query)
            articles = response.get("articles", [])
        except HTTPError as exc:
            exception_body = exc.response.json()
            logging.error(f'NewsAPI failed for {query}: {exception_body}')
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
                logging.error(
                    f'Get Binance exchange_info failed for {currency["exchange_symbol"]}: {exception_body}'
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
        self, trending_coins, coindesk_articles, cryptopanic_posts
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

        coin_info = await self._filter_conversion_currency(coin_info)
        return await self._filter_currencies_by_binance(coin_info)

    async def _get_metrics_per_currency(
        self, coin_info, cryptopanic_posts
    ):
        rows: List[dict[str, float | str]] = []
        for info in coin_info:
            symbol = info['symbol']
            name = info['name']
            coingecko_id = self.symbol_to_coingecko_id.get(symbol)

            comm_score = await self._get_coin_sentiment(coingecko_id)
            await sleep(1)

            news_score = self.get_news_sentiment(name)

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

    async def _generate_massage(self, df):
        results = df.sort_values("composite", ascending=False)[["symbol", "composite", "signal"]]
        message_lines = [
            f"{row['symbol']}" + ((12 - len(row['symbol'])) * ' ') + f"{row['composite']:.3f} ({row['signal']})"
            for _, row in results.iterrows()
        ]
        return '```\n' + "\n".join(message_lines) + '```'

    async def process_trending_currencies(self, context: ContextTypes.DEFAULT_TYPE):
        try:
            trending_coins = self._get_trending_coins()
            coindesk_articles = self._get_coindesk_articles(limit=10)
            cryptopanic_posts = self._get_cryptopanic_posts(limit=50)

            coin_info = await self._forming_currencies_list(
                trending_coins, coindesk_articles, cryptopanic_posts
            )

            rows = await self._get_metrics_per_currency(coin_info, cryptopanic_posts)

            df = await self._calculate_signals(rows)

            message = await self._generate_massage(df)
            print(message)
            await context.bot.send_message(self.chat_id, message, parse_mode=ParseMode.MARKDOWN_V2)

        except CmcException as ex:
            print('CmcException ex', ex)
            # self.sentry_client.capture_exception(ex)
        except BaseException as ex:
            print('BaseException ex', ex)
            # self.sentry_client.capture_exception(ex)
