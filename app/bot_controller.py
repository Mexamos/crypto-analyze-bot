import logging
import json
import pickle
from asyncio import sleep as async_sleep
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from functools import wraps
from time import sleep as sync_sleep
from decimal import Decimal

import pandas as pd
from pytz import timezone
from requests.exceptions import HTTPError
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, ConversationHandler, CallbackQueryHandler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from prophet import Prophet
from sentry_sdk import push_scope
import numpy as np

from app.cache.client import CacheClient
from app.database.client import DatabaseClient
from app.crypto.coindesk_client import CoindeskClient
from app.crypto.coingecko_client import CoingeckoClient
from app.crypto.coinmarketcap_client import CoinmarketcapClient
from app.crypto.cryptopanic_client import CryptopanicClient
from app.crypto.binance_client import BinanceClient
from app.crypto.newsapi_client import NewsapiClient
from app.crypto.santimentapi_client import SantimentApiClient
from app.crypto.santimentapi_model import ModelTrainingFacade, ModelPredictionFacade
from app.config import Config
from app.monitoring.sentry import SentryClient
from app.processors.analyze_processor import AnalyzeProcessor

SIMULATE_PURCHASE_CHOOSE_CURRENCY = 0


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
        coinmarketcap_client: CoinmarketcapClient,
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
        self.coinmarketcap_client = coinmarketcap_client
        self.model_training_facade = model_training_facade
        self.model_prediction_facade = model_prediction_facade

        self.cache_client: CacheClient = cache_client
        self.sentry_client = sentry_client
        self.config = config
        self.logger: logging.Logger = logging.getLogger(__name__)

        self.symbol_to_coingecko_id = {}
        self.timezone = timezone(self.config.timezone_name)
        self.date_format = self.config.common_date_format

        self.bot_token = token
        self.chat_ids = [int(chat_id) for chat_id in chat_ids]
        self.stopped = False

    def init_bot(self):
        self.app = Application.builder().token(self.bot_token).build()

        self.app.add_handler(CommandHandler("health", self.health))
        self.app.add_handler(CommandHandler("scikit_learn_predict", self.scikit_learn_predict))
        self.app.add_handler(CommandHandler("stop", self.stop))

        job_queue = self.app.job_queue
        self.analyze_processor = AnalyzeProcessor(
            binance_client=self.binance_client,
            coinmarketcap_client=self.coinmarketcap_client,
            sentry_client=self.sentry_client,
            config=self.config,
            chat_ids=self.chat_ids,
        )
        job_queue.run_repeating(
            self.analyze_processor.process, interval=self.config.analyze_task_interval, first=1
        )
        self.app.add_handler(CommandHandler("get_price_changes", self.analyze_processor.get_price_changes))

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
        self.stopped = self.trending_processor.stopped = True
        await update.message.reply_text('Bot is stopping')

    async def health(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message.chat_id not in self.chat_ids:
            return

        jobs = context.job_queue.jobs()

        await update.message.reply_text(f'number_running_tasks={len(jobs)}')

    async def validate_currency(self, currency: str, projects: pd.DataFrame) -> None:
        currency_lower = currency.lower()
        exists = ((
            projects['ticker'].str.lower() == currency_lower
        ) | (
            projects['slug'].str.lower() == currency_lower
        )).any()

        if not exists:
            raise Exception(f"Invalid currency: {currency}")

    async def get_slug_by_currency(self, currency: str, projects: pd.DataFrame) -> str:
        currency_lower = currency.lower()
        slug = projects.loc[
            (projects['ticker'].str.lower() == currency_lower) |
            (projects['slug'].str.lower() == currency_lower),
            'slug'
        ].iloc[0]

        if not slug:
            raise ValueError(f"Currency '{currency}' not found in projects.")

        return slug

    async def scikit_learn_predict(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or update.message.chat_id not in self.chat_ids:
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

            projects = self.santiment_api_client.get_all_projects()
            await self.validate_currency(currency_name, projects)
            slug = await self.get_slug_by_currency(currency_name, projects)

            self.model_training_facade.train_and_save_model(
                [slug], start_training_date, end_training_date
            )
            message = self.model_prediction_facade.predict([slug])

            await update.message.reply_text(message)

        except Exception as ex:
            self.sentry_client.capture_exception(ex)
            await update.message.reply_text(f'Error in scikit_learn_predict: {ex}')
