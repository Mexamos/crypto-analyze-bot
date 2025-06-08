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
from app.crypto.cryptopanic_client import CryptopanicClient
from app.crypto.binance_client import BinanceClient
from app.crypto.newsapi_client import NewsapiClient
from app.crypto.santimentapi_client import SantimentApiClient
from app.crypto.santimentapi_model import ModelTrainingFacade, ModelPredictionFacade
from app.config import Config
from app.monitoring.sentry import SentryClient
from app.trending_processor import TrendingCurrencyProcessor

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
        # self.app.add_handler(CommandHandler("get_config", self.get_config))
        # self.app.add_handler(CommandHandler("change_config", self.change_config))
        self.app.add_handler(CommandHandler("get_last_trending_currencies", self.get_last_trending_currencies))
        self.app.add_handler(CommandHandler("scikit_learn_predict", self.scikit_learn_predict))
        self.app.add_handler(CommandHandler("stop", self.stop))

        # conv_handler = ConversationHandler(
        #     entry_points=[CommandHandler("simulate_purchase", self.simulate_purchase_start)],
        #     states={
        #         SIMULATE_PURCHASE_CHOOSE_CURRENCY: [
        #             CallbackQueryHandler(self.simulate_purchase_choice)
        #         ],
        #     },
        #     fallbacks=[],
        # )
        # self.app.add_handler(conv_handler)

        job_queue = self.app.job_queue
        self.trending_processor = TrendingCurrencyProcessor(
            coindesk_client=self.coindesk_client,
            coingecko_client=self.coingecko_client,
            cryptopanic_client=self.cryptopanic_client,
            newsapi_client=self.newsapi_client,
            santiment_api_client=self.santiment_api_client,
            binance_client=self.binance_client,
            model_training_facade=self.model_training_facade,
            model_prediction_facade=self.model_prediction_facade,
            cache_client=self.cache_client,
            sentry_client=self.sentry_client,
            config=self.config,
            chat_ids=self.chat_ids,
            symbol_to_coingecko_id=self.symbol_to_coingecko_id,
        )
        job_queue.run_repeating(
            self.trending_processor.process, interval=self.config.analyze_task_interval, first=1
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
        self.stopped = self.trending_processor.stopped = True
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

    async def get_last_trending_currencies(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message.chat_id not in self.chat_ids:
            return

        cached_records = self.cache_client.get('last_trending_currencies')
        if not cached_records:
            return

        last_records = json.loads(cached_records)
        records = []
        for record in last_records:
            record_parts = record.split(':')
            records.append({
                'symbol': record_parts[0],
                'exchange_symbol': record_parts[1],
                'name': record_parts[2]
            })

        if not records:
            return None

        message_lines = [
            f"{record['symbol']}" + ((12 - len(record['symbol'])) * ' ') + f"{record['name']}"
            for record in records
        ]
        message = '```\n' + "\n".join(message_lines) + '```'
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN_V2)

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

    # async def _get_historical_price(self, symbol: str, purchase_date: datetime) -> Optional[Decimal]:
    #     """Get historical price for a symbol at a specific date"""
    #     try:
    #         # Convert datetime to timestamp for Binance API
    #         timestamp = int(purchase_date.timestamp() * 1000)
            
    #         # Get klines data for the specific time
    #         klines = self.binance_client.klines(
    #             symbol=symbol,
    #             interval='1h',
    #             start_time=timestamp,
    #             end_time=timestamp + 3600000,  # Add 1 hour to ensure we get data
    #             limit=1
    #         )
            
    #         if klines and len(klines) > 0:
    #             # Return the closing price
    #             return Decimal(str(klines[0][4]))
    #         return None
    #     except Exception as e:
    #         self.sentry_client.capture_exception(e)
    #         return None

    # async def _get_current_price(self, symbol: str) -> Optional[Decimal]:
    #     """Get current price for a symbol"""
    #     try:
    #         # Get the latest kline
    #         klines = self.binance_client.klines(
    #             symbol=symbol,
    #             interval='1h',
    #             limit=1
    #         )
            
    #         if klines and len(klines) > 0:
    #             return Decimal(str(klines[0][4]))
    #         return None
    #     except Exception as e:
    #         self.sentry_client.capture_exception(e)
    #         return None

    # async def _find_matching_symbols(self, symbol: str) -> List[str]:
    #     """Find all matching symbols in Binance"""
    #     try:
    #         exchange_info = self.binance_client.exchange_info(symbol)
    #         if exchange_info and 'symbols' in exchange_info:
    #             return [s['symbol'] for s in exchange_info['symbols'] if s['symbol'].startswith(symbol)]
    #         return []
    #     except Exception as e:
    #         self.sentry_client.capture_exception(e)
    #         return []

    # async def simulate_purchase_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    #     if update.message.chat_id not in self.chat_ids:
    #         return ConversationHandler.END

    #     options = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
    #     keyboard = [
    #         [InlineKeyboardButton(opt, callback_data=opt)]
    #         for opt in options
    #     ]
    #     reply_markup = InlineKeyboardMarkup(keyboard)

    #     await update.message.reply_text(
    #         "Пожалуйста, выберите символ для симуляции покупки:",
    #         reply_markup=reply_markup
    #     )
    #     return SIMULATE_PURCHASE_CHOOSE_CURRENCY

    # async def simulate_purchase_choice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    #     query = update.callback_query
    #     await query.answer()

    #     chosen_symbol = query.data
    #     print(f"Chosen symbol: {chosen_symbol}, type: {type(chosen_symbol)}")
    #     # Save the chosen symbol in user_data
    #     context.user_data['simulate_symbol'] = chosen_symbol

    #     # Remove the inline keyboard
    #     await query.edit_message_reply_markup(reply_markup=None)

    #     # # Тут можно повторить всё, что было в оригинальном simulate_purchase,

    #     return ConversationHandler.END



    # async def simulate_purchase(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    #     if update.message.chat_id not in self.chat_ids:
    #         return

    #     if not context.args:
    #         await update.message.reply_text(
    #             'Please provide a cryptocurrency symbol (e.g., /simulate_purchase BTC)'
    #         )
    #         return

    #     symbol = context.args[0].upper()
    #     purchase_date = None
    #     if len(context.args) > 1:
    #         try:
    #             try:
    #                 purchase_date = datetime.strptime(context.args[1], '%Y-%m-%d %H:%M')
    #             except ValueError:
    #                 purchase_date = datetime.strptime(context.args[1], self.date_format)
    #         except ValueError:
    #             await update.message.reply_text(
    #                 'Invalid date format. Please use YYYY-MM-DD HH:MM or YYYY-MM-DD'
    #             )
    #             return

    #     matching_symbols = await self._find_matching_symbols(symbol)
        
    #     if not matching_symbols:
    #         await update.message.reply_text(f'No matching symbols found for {symbol}')
    #         return
        
    #     if len(matching_symbols) > 1:
    #         symbols_list = '\n'.join(matching_symbols)
    #         await update.message.reply_text(
    #             f'Multiple matching symbols found. Please choose one:\n{symbols_list}'
    #         )
    #         return

    #     # Get the exact symbol
    #     exact_symbol = matching_symbols[0]

    #     # Check if we already have a purchase record
    #     existing_purchase = self.db_client.find_cryptocurrency_purchase_by_symbol(exact_symbol)
    #     if existing_purchase:
    #         await update.message.reply_text(
    #             f'You have already purchased {exact_symbol} ({existing_purchase.name}) '
    #             f'at {existing_purchase.date.strftime("%Y-%m-%d %H:%M")} '
    #             f'for ${existing_purchase.price}'
    #         )
    #         return

    #     # Get price
    #     price = None
    #     if purchase_date:
    #         price = await self._get_historical_price(exact_symbol, purchase_date)
    #     else:
    #         price = await self._get_current_price(exact_symbol)

    #     if not price:
    #         await update.message.reply_text(f'Could not get price for {exact_symbol}')
    #         return

    #     # Get cryptocurrency name and coingecko_id
    #     name = exact_symbol
    #     coingecko_id = None
    #     try:
    #         exchange_info = self.binance_client.exchange_info(exact_symbol)
    #         if exchange_info and 'symbols' in exchange_info:
    #             for symbol_info in exchange_info['symbols']:
    #                 if symbol_info['symbol'] == exact_symbol:
    #                     name = symbol_info.get('baseAsset', exact_symbol)
    #                     break
    #     except Exception as e:
    #         self.sentry_client.capture_exception(e)

    #     # Try to get coingecko_id from our mapping
    #     coingecko_id = self.symbol_to_coingecko_id.get(name.upper())

    #     # Create purchase record
    #     purchase_date = purchase_date or datetime.now()
    #     self.db_client.create_cryptocurrency_purchase(
    #         symbol=exact_symbol,
    #         name=name,
    #         price=price,
    #         date=purchase_date,
    #         coingecko_id=coingecko_id
    #     )

    #     # Send confirmation message
    #     date_str = purchase_date.strftime('%Y-%m-%d %H:%M')
    #     await update.message.reply_text(
    #         f'Successfully simulated purchase of {exact_symbol} ({name}) at {date_str} for ${price}'
    #     )
