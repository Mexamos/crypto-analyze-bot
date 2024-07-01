import math
from datetime import datetime
from decimal import Decimal
from typing import List, Set

import pandas as pd
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes
from pytz import timezone

from app.database.client import DatabaseClient
from app.database.models import CurrencyPrice
from app.crypto.coinmarketcap_client import CoinmarketcapClient, CmcException
from app.crypto.binance_client import BinanceClient
from app.analytics.chart import ChartController, DataForChartNotFound, DataForIncomesTableNotFound
from app.config import Config
from app.analytics.google_sheets_client import GoogleSheetsClient, GoogleSheetAppendIncomeFailed
from app.utils import scientific_notation_to_usual_format
from app.monitoring.sentry import SentryClient


class BotController:

    def __init__(
        self, db_client: DatabaseClient, cmc_client: CoinmarketcapClient, binance_cleint: BinanceClient,
        chart_controller: ChartController, google_sheets_client: GoogleSheetsClient,
        sentry_client: SentryClient, config: Config, token: str, chat_id: str
    ) -> None:
        self.db_client = db_client
        self.cmc_client = cmc_client
        self.binance_cleint = binance_cleint
        self.chart_controller = chart_controller
        self.google_sheets_client = google_sheets_client
        self.config = config
        self.sentry_client = sentry_client

        self.bot_token = token
        self.chat_id = int(chat_id)

        self.known_currencies = set()

        self.out_of_trend_currencies = set()

        self.timezone = timezone(self.config.timezone_name)
        self.launch_datetime = datetime.now(self.timezone)

        self.stop_buying_flag = False

    def init_bot(self):
        self.app = Application.builder().token(self.bot_token).build()

        self.app.add_handler(CommandHandler("prices_on_chart", self.prices_on_chart))
        self.app.add_handler(CommandHandler("incomes_table", self.incomes_table))
        self.app.add_handler(CommandHandler("list_current_currencies", self.list_current_currencies))
        self.app.add_handler(CommandHandler("list_income_currencies", self.list_income_currencies))
        self.app.add_handler(CommandHandler("list_out_of_trend_currencies", self.list_out_of_trend_currencies))

        self.app.add_handler(CommandHandler("health", self.health))
        self.app.add_handler(CommandHandler("get_config", self.get_config))
        self.app.add_handler(CommandHandler("change_config", self.change_config))
        self.app.add_handler(CommandHandler("stop_buying", self.stop_trading))
        self.app.add_handler(CommandHandler("start_buying", self.start_trading))
        self.app.add_handler(CommandHandler("stop", self.stop))

        job_queue = self.app.job_queue
        job_queue.run_repeating(self.process_trending_currencies, interval=self.config.process_trending_currencies_interval)

    def restore_unsold_currencies(self):
        rows = self.google_sheets_client.get_unsold_currencies()
        for row in rows:
            self._create_currency_price(
                cmc_id=int(row[2]),
                symbol=row[1],
                price=Decimal(row[3]),
                date_time=datetime.strptime(row[0], "%d.%m.%Y %H:%M:%S"),
            )

        self.google_sheets_client.delete_unsold_currencies()

    def run_bot(self):
        # Run the bot until the user presses Ctrl-C
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)

    async def stop_trading(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message.chat_id != self.chat_id:
            return

        self.stop_buying_flag = True

    async def start_trading(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message.chat_id != self.chat_id:
            return

        self.stop_buying_flag = False

    async def stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message.chat_id != self.chat_id:
            return

        jobs =  context.job_queue.jobs()
        for job in jobs:
            job.schedule_removal()

        await self._sell_or_record_to_table_rest_currencies(context)

        await self._generate_incomes_report()

        self.app.stop_running()
        await update.message.reply_text('Bot stopped')

    async def _sell_or_record_to_table_rest_currencies(self, context: ContextTypes.DEFAULT_TYPE):
        bought_currencies = self.db_client.find_currency_price_cmc_ids()

        for cmc_id in bought_currencies:
            currency_prices = self.db_client.find_currency_price_by_cmc_id(cmc_id)
            if len(currency_prices) == 0:
                continue

            first = currency_prices[0]
            last = currency_prices[len(currency_prices) - 1]

            if last.price >= first.price:
                await self._sell_currency(context, last.cmc_id, last.symbol, last.price)
            else:
                await self._record_unsold_currency(first)

    async def _record_unsold_currency(self, currency_price: CurrencyPrice):
        self.google_sheets_client.append_unsold_currency(
            currency_price.date_time, currency_price.cmc_id,
            currency_price.symbol, currency_price.price
        )

    async def _generate_incomes_report(self):
        incomes = self.db_client.find_income_sum_by_cmc_id()

        currencies = []
        income_amount = 0
        for income in incomes:
            currencies.append(str(income[0]))
            income_amount += income[1]

        currencies_string = ', '.join(currencies)

        self.google_sheets_client.append_incomes_report(
            self.launch_datetime, datetime.now(self.timezone),
            currencies_string, scientific_notation_to_usual_format(income_amount)
        )

    async def health(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message.chat_id != self.chat_id:
            return

        jobs = context.job_queue.jobs()

        currency_prices_count = self.db_client.count_all_currency_price()

        self.google_sheets_client.append_to_test_connection(datetime.now(self.timezone))
        google_sheet_link = f'https://docs.google.com/spreadsheets/d/{self.google_sheets_client.spreadsheet_id}/'

        coin_market_cap_latest_request_datetime = (
            self.cmc_client.latest_request_datetime.strftime("%d.%m.%Y %H:%M:%S")
            if self.cmc_client.latest_request_datetime else None
        )

        await update.message.reply_text(f'number_running_tasks={len(jobs)}')
        await update.message.reply_text(f'currency_prices_count={currency_prices_count}')
        await update.message.reply_text(f'google_sheet_link={google_sheet_link}')
        await update.message.reply_text(f'coin_market_cap_latest_request_datetime={coin_market_cap_latest_request_datetime}')

    async def get_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message.chat_id != self.chat_id:
            return

        parameter_list = '\n'.join([f'{name}={getattr(self.config, name)}' for name in self.config.parameter_list])
        await update.message.reply_text(parameter_list)

    async def change_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message.chat_id != self.chat_id:
            return

        if len(context.args) < 2:
            await update.message.reply_text('Need config parameter name and new value as comand arguments!')
            return

        try:
            self.config.change_value(context.args[0], context.args[1])
        except BaseException as ex:
            self.sentry_client.capture_exception(ex)

    async def prices_on_chart(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message.chat_id != self.chat_id:
            return

        if len(context.args) == 0:
            await update.message.reply_text('Need blockchain symbol as comand argument!')
            return

        symbol = context.args[0].upper()
        try:
            plot_file = self.chart_controller.generate_chart_image(symbol)
            await update.message.reply_photo(plot_file)
        except DataForChartNotFound as ex:
            self.sentry_client.capture_exception(ex)

    async def incomes_table(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message.chat_id != self.chat_id:
            return

        symbol = context.args[0].upper() if len(context.args) > 0 else None
        try:
            table_file = self.chart_controller.generate_incomes_table(symbol)
            await update.message.reply_photo(table_file)
        except DataForIncomesTableNotFound as ex:
            self.sentry_client.capture_exception(ex)

    async def list_current_currencies(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message.chat_id != self.chat_id:
            return

        currency_prices = self.db_client.find_first_currency_prices_grouped_by_cmc_id()
        if len(currency_prices) > 0:
            currency_prices = [
                f'{cp[2]}' + ((10 - len(cp[2])) * ' ') + 
                f'{datetime.strptime(cp[4], "%Y-%m-%d %H:%M:%S.%f").strftime("%d.%m.%Y %H:%M:%S")}'
                for cp in currency_prices
            ]
            result_string = '```\n' + '\n'.join(currency_prices) + '```'
            await update.message.reply_text(result_string, parse_mode=ParseMode.MARKDOWN_V2)
        else:
            await update.message.reply_text('Currency price data not found')

    async def list_income_currencies(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message.chat_id != self.chat_id:
            return

        currencies = self.db_client.find_income_symbols()
        if len(currencies) > 0:
            await update.message.reply_text('\n'.join(currencies))
        else:
            await update.message.reply_text('Income data not found')

    async def list_out_of_trend_currencies(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message.chat_id != self.chat_id:
            return

        if len(self.out_of_trend_currencies) > 0:
            await update.message.reply_text('\n'.join(list(self.out_of_trend_currencies)))
        else:
            await update.message.reply_text('Out of trend currencies not found')

    async def _filter_currencies_by_binance(self, currencies: List[dict]) -> List[dict]:
        available_currencies = []
        for currency in currencies:
            exchange_infos = self.binance_cleint.find_exchange_info(
                toAsset=currency['symbol'], fromAsset=self.config.currency_conversion
            )
            if isinstance(exchange_infos, list) and len(exchange_infos) > 0:
                available_currencies.append(currency)

        return available_currencies

    async def _filter_conversion_currency(self, currencies: List[dict]) -> List[dict]:
        available_currencies = []
        for currency in currencies:
            if currency['symbol'] != self.config.currency_conversion:
                available_currencies.append(currency)

        return available_currencies

    async def _is_funds_to_buy_new_currency(self) -> bool:
        number_available_currencies = math.floor(self.config.total_available_amount / self.config.transactions_amount)
        bought_currencies = self.db_client.find_currency_price_cmc_ids()

        return len(bought_currencies) < number_available_currencies

    async def _is_there_buy_signal(self, cmc_id: int, last_updated, last_price):
        data = self.cmc_client.get_quotes_historical(cmc_id)
        dates = [quote['timestamp'] for quote in data['data']['quotes']] + [last_updated]
        prices = [quote['quote']['USD']['price'] for quote in data['data']['quotes']] + [last_price]
        data_frame_input = {
            'Date': dates,
            'Price': prices
        }
        print('data_frame_input', data_frame_input)

        df = pd.DataFrame(data_frame_input)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Вычисление скользящей средней
        window = self.config.quotes_historical_count
        df['SMA'] = df['Price'].rolling(window=window).mean()

        # Вычисление моментума
        df['Momentum'] = df['Price'] - df['Price'].shift(window)

        # Вычисление ROC
        df['ROC'] = ((df['Price'] - df['Price'].shift(window)) / df['Price'].shift(window)) * 100

        print('prices[-1]', prices[-1])
        print("df['SMA'].iloc[-1]", df['SMA'].iloc[-1])
        print("df['Momentum'].iloc[-1]", df['Momentum'].iloc[-1])
        print("df['ROC'].iloc[-1]", df['ROC'].iloc[-1])
        print("prices[-1] > df['SMA'].iloc[-1]", prices[-1] > df['SMA'].iloc[-1])
        print("df['Momentum'].iloc[-1] > 0", df['Momentum'].iloc[-1] > 1)
        print("df['ROC'].iloc[-1] > 0", df['ROC'].iloc[-1] > 1)
        if (
            prices[-1] > df['SMA'].iloc[-1] and
            df['Momentum'].iloc[-1] > 1 and
            df['ROC'].iloc[-1] > 1
        ):
            return True
        else:
            return False

    async def _is_there_sell_signal(self, cmc_id: int, last_updated, last_price):
        data = self.cmc_client.get_quotes_historical(cmc_id)
        dates = [quote['timestamp'] for quote in data['data']['quotes']] + [last_updated]
        prices = [quote['quote']['USD']['price'] for quote in data['data']['quotes']] + [last_price]
        data_frame_input = {
            'Date': dates,
            'Price': prices
        }
        print('data_frame_input', data_frame_input)

        df = pd.DataFrame(data_frame_input)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Вычисление скользящей средней
        window = self.config.quotes_historical_count
        df['SMA'] = df['Price'].rolling(window=window).mean()

        # Вычисление моментума
        df['Momentum'] = df['Price'] - df['Price'].shift(window)

        # Вычисление ROC
        df['ROC'] = ((df['Price'] - df['Price'].shift(window)) / df['Price'].shift(window)) * 100

        print('prices[-1]', prices[-1])
        print("df['SMA'].iloc[-1]", df['SMA'].iloc[-1])
        print("df['Momentum'].iloc[-1]", df['Momentum'].iloc[-1])
        print("df['ROC'].iloc[-1]", df['ROC'].iloc[-1])
        print("prices[-1] > df['SMA'].iloc[-1]", prices[-1] < df['SMA'].iloc[-1])
        print("df['Momentum'].iloc[-1] > 0", df['Momentum'].iloc[-1] < -1)
        print("df['ROC'].iloc[-1] > 0", df['ROC'].iloc[-1] < -1)
        if (
            prices[-1] < df['SMA'].iloc[-1] and
            df['Momentum'].iloc[-1] < -1 and
            df['ROC'].iloc[-1] < -1
        ):
            return True
        else:
            return False

    def _create_currency_price(self, cmc_id: int, symbol: str, price, date_time: datetime):
        self.db_client.create_currency_price(
            cmc_id=cmc_id,
            symbol=symbol,
            price=price,
            date_time=date_time,
        )

    async def _create_income(self, cmc_id: int, price: Decimal):
        currency_prices = self.db_client.find_currency_price_by_cmc_id(cmc_id)
        first_currency_price = currency_prices[0]

        first_price = first_currency_price.price
        income_value = ((self.config.transactions_amount / first_price) * price) - self.config.transactions_amount

        first_date_time = first_currency_price.date_time
        last_date_time = datetime.now(self.timezone)

        self.db_client.create_income(
            cmc_id=cmc_id,
            symbol=first_currency_price.symbol,
            value=income_value,
            date_time=last_date_time,
        )

        self.google_sheets_client.append_income(
            first_date_time, last_date_time, float(first_price), float(price),
            first_currency_price.symbol, float(income_value)
        )

    async def _sell_currency(self, context: ContextTypes.DEFAULT_TYPE, cmc_id: int, symbol: str, price: Decimal):
        try:
            # TODO add convert to self.config.currency_conversion request

            # # Send result prices chart
            # chart_file = self.chart_controller.generate_chart_image(symbol)
            # await context.bot.send_photo(self.chat_id, chart_file)

            # Add new income data
            await self._create_income(cmc_id, price)

            # Delete existing currecny prices
            self.db_client.delete_currency_prices(cmc_id)
        except (DataForChartNotFound, GoogleSheetAppendIncomeFailed) as ex:
            self.sentry_client.capture_exception(ex)
        except BaseException as ex:
            self.sentry_client.capture_exception(ex)

    async def _sell_currency_without_profit(self, context: ContextTypes.DEFAULT_TYPE, cmc_id: int):
        currency_prices = self.db_client.find_currency_price_by_cmc_id(cmc_id)
        if len(currency_prices) == 0:
            return True

        first = currency_prices[0]

        currency_data = self.cmc_client.get_quotes_latest(first.cmc_id)
        latest_price = Decimal(str(currency_data['data'][str(first.cmc_id)]['quote']['USD']['price']))
        self._create_currency_price(
            cmc_id=first.cmc_id, symbol=first.symbol, price=latest_price,
            date_time=datetime.now(self.timezone),
        )

        await self._sell_currency(context, first.cmc_id, first.symbol, latest_price)

    async def _add_data(self, currencies: List[dict], bought_currencies: Set[int]) -> None:
        for currency in currencies:
            cmc_id = currency['id']
            symbol = currency['symbol']
            price = Decimal(str(currency['quote']['USD']['price']))

            if cmc_id in bought_currencies:
                self._create_currency_price(
                    cmc_id=cmc_id, symbol=symbol, price=price,
                    date_time=datetime.now(self.timezone),
                )

    async def _sell(
        self, context: ContextTypes.DEFAULT_TYPE, bought_currencies: Set[int]
    ) -> None:
        for cmc_id in bought_currencies:
            currency = self.cmc_client.get_quotes_latest(cmc_id)

            symbol = currency['data'][str(cmc_id)]['symbol']
            price = Decimal(str(currency['data'][str(cmc_id)]['quote']['USD']['price']))

            if (
                await self._is_there_sell_signal(
                    cmc_id, currency['data'][str(cmc_id)]['last_updated'],
                    currency['data'][str(cmc_id)]['quote']['USD']['price']
                )
            ):
                await self._sell_currency(context, cmc_id, symbol, price)

    async def _sell_without_profit(
        self, context: ContextTypes.DEFAULT_TYPE, currencies: List[dict], bought_currencies: Set[int]
    ) -> None:
        # update out of trends
        for currency in currencies:
            cmc_id = currency['id']

            if cmc_id in bought_currencies:
                bought_currencies.remove(cmc_id)

            if cmc_id in self.out_of_trend_currencies:
                self.out_of_trend_currencies.remove(cmc_id)

        self.out_of_trend_currencies.update(bought_currencies)

        # sell
        sold_currencies = set()
        for cmc_id in self.out_of_trend_currencies:
            await self._sell_currency_without_profit(context, cmc_id)
            sold_currencies.add(cmc_id)

        self.out_of_trend_currencies.difference_update(sold_currencies)

    async def _buy(self, currencies: List[dict], bought_currencies: Set[int]) -> None:
        for currency in currencies:
            cmc_id = currency['id']
            symbol = currency['symbol']
            price = Decimal(str(currency['quote']['USD']['price']))

            if (
                cmc_id not in bought_currencies and
                await self._is_funds_to_buy_new_currency() and
                await self._is_there_buy_signal(
                    cmc_id, currency['last_updated'], currency['quote']['USD']['price']
                ) and
                not self.stop_buying_flag
            ):
                self._create_currency_price(
                    cmc_id=cmc_id, symbol=symbol, price=price,
                    date_time=datetime.now(self.timezone),
                )

    async def process_trending_currencies(self, context: ContextTypes.DEFAULT_TYPE):
        try:
            print('process_trending_currencies 1')
            currencies = self.cmc_client.get_trending_currencies()
            print('process_trending_currencies 2')
            currencies = await self._filter_currencies_by_binance(currencies)
            print('process_trending_currencies 3')
            currencies = await self._filter_conversion_currency(currencies)
            print('process_trending_currencies 4')

            bought_currencies = set(self.db_client.find_currency_price_cmc_ids())
            await self._add_data(currencies, bought_currencies)
            print('process_trending_currencies 5')

            bought_currencies = set(self.db_client.find_currency_price_cmc_ids())
            await self._sell(context, bought_currencies)
            print('process_trending_currencies 6')

            await self._buy(currencies, bought_currencies)
            print('process_trending_currencies 7')

        except CmcException as ex:
            print('ex', ex)
            self.sentry_client.capture_exception(ex)
        except BaseException as ex:
            print('ex', ex)
            self.sentry_client.capture_exception(ex)
