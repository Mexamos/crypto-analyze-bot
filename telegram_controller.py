import math
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes
from pytz import timezone

from database.client import DatabaseClient
from coinmarketcap_client import CoinmarketcapClient, CmcException
from binance_client import BinanceClient, BinanceException
from chart import ChartController, DataForChartNotFound, DataForIncomesTableNotFound
from config import Config
from google_sheets_client import GoogleSheetsClient, GoogleSheetAppendIncomeFailed


class RequestCurrenciesAction(Enum):
    BUY = 1
    SELL = 2
    ADD_DATA = 3
    SKIP = 4


class TelegramController:

    def __init__(
        self, db_client: DatabaseClient, cmc_client: CoinmarketcapClient, binance_cleint: BinanceClient,
        chart_controller: ChartController, google_sheets_client: GoogleSheetsClient,
        config: Config, token: str, chat_id: str
    ) -> None:
        self.db_client = db_client
        self.cmc_client = cmc_client
        self.binance_cleint = binance_cleint
        self.chart_controller = chart_controller
        self.google_sheets_client = google_sheets_client
        self.config = config
        self.app = Application.builder().token(token).build()
        self.chat_id = int(chat_id)

        self.timezone = timezone(self.config.timezone_name)

        self.app.add_handler(CommandHandler("prices_on_chart", self.prices_on_chart))
        self.app.add_handler(CommandHandler("incomes_table", self.incomes_table))
        self.app.add_handler(CommandHandler("list_current_currencies", self.list_current_currencies))
        self.app.add_handler(CommandHandler("list_income_currencies", self.list_income_currencies))

        self.app.add_handler(CommandHandler("get_config", self.get_config))
        self.app.add_handler(CommandHandler("change_config", self.change_config))
        self.app.add_handler(CommandHandler("force_stop", self.force_stop))

        job_queue = self.app.job_queue
        job_queue.run_repeating(self.request_currencies, interval=self.config.request_currencies_interval)

    def run_bot(self):
        # Run the bot until the user presses Ctrl-C
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)

    async def force_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self.app.stop_running()
        await update.message.reply_text('Bot is stopped')

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
            await update.message.reply_text(f'Exception: {str(ex)}')

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
            await update.message.reply_text(str(ex))

    async def incomes_table(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message.chat_id != self.chat_id:
            return

        symbol = context.args[0].upper() if len(context.args) > 0 else None
        try:
            table_file = self.chart_controller.generate_incomes_table(symbol)
            await update.message.reply_photo(table_file)
        except DataForIncomesTableNotFound as ex:
            await update.message.reply_text(str(ex))

    async def list_current_currencies(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message.chat_id != self.chat_id:
            return

        currency_prices = self.db_client._find_first_currency_prices_grouped_by_symbol()
        if len(currency_prices) > 0:
            currency_prices = [
                f'{cp[1]}' + ((10 - len(cp[1])) * ' ') + 
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

    async def _is_funds_buy_new_currency(self, currency: dict) -> bool:
        number_available_currencies = math.floor(self.config.total_available_amount / self.config.transactions_amount)
        bought_currencies = self.db_client.find_currency_price_symbols()

        return len(bought_currencies) < number_available_currencies

    async def _define_action(self, currency: dict) -> RequestCurrenciesAction:
        currency_prices = self.db_client.find_currency_price_by_symbol(currency['symbol'])

        if len(currency_prices) == 0:
            if await self._is_funds_buy_new_currency(currency):
                return RequestCurrenciesAction.BUY
            else:
                return RequestCurrenciesAction.SKIP

        currency_price = currency_prices[0]
        old_price = currency_price.price
        new_price = Decimal(str(currency['quote']['USD']['price']))

        difference = new_price - old_price

        absolute_difference = abs(old_price - new_price)
        sum_of_values = (old_price + new_price) / 2
        difference_in_percentage = (absolute_difference / sum_of_values) * 100

        if difference <= 0 or difference_in_percentage < self.config.percentage_difference_for_sale:
            return RequestCurrenciesAction.ADD_DATA

        return RequestCurrenciesAction.SELL

    async def _create_currency_price(self, currency: dict):
        self.db_client.create_currency_price(
            symbol=currency['symbol'],
            price=currency['quote']['USD']['price'],
            percent_change_24h=currency['quote']['USD']['percent_change_24h'],
            date_time=datetime.now(self.timezone),
        )

    async def caclculate_income_value(self, currency: dict) -> Decimal:
        currency_prices = self.db_client.find_currency_price_by_symbol(currency['symbol'])
        currency_price = currency_prices[0]

        old_price = currency_price.price
        new_price = Decimal(str(currency['quote']['USD']['price']))

        return (new_price * self.config.transactions_amount) - (old_price * self.config.transactions_amount)

    async def _create_income(self, currency: dict):
        value = await self.caclculate_income_value(currency)
        date_time = datetime.now(self.timezone)
        self.db_client.create_income(
            symbol=currency['symbol'],
            value=value,
            date_time=date_time,
        )

        self.google_sheets_client.append_income_row(date_time, currency['symbol'], str(value))

    async def _sell_currency(self, context: ContextTypes.DEFAULT_TYPE, currency: dict):
        symbol = currency['symbol']
        try:
            # TODO add convert to self.config.currency_conversion request

            # Send result prices chart
            chart_file = self.chart_controller.generate_chart_image(symbol)
            await context.bot.send_photo(self.chat_id, chart_file)

            # Add new income data
            await self._create_income(currency)

            # Delete existing currecny prices
            self.db_client.delete_currency_prices(symbol)
        except (DataForChartNotFound, GoogleSheetAppendIncomeFailed) as ex:
            await context.bot.send_message(self.chat_id, str(ex))

    async def request_currencies(self, context: ContextTypes.DEFAULT_TYPE):
        try:
            currencies = self.cmc_client.actual_trending_latest_currencies()
            currencies = await self._filter_currencies_by_binance(currencies)
            currencies = await self._filter_conversion_currency(currencies)

            for currency in currencies:
                action = await self._define_action(currency)

                if action == RequestCurrenciesAction.SKIP:
                    continue

                await self._create_currency_price(currency)
                if action in (RequestCurrenciesAction.BUY, RequestCurrenciesAction.ADD_DATA):
                    pass
                if action == RequestCurrenciesAction.SELL:
                    await self._sell_currency(context, currency)

        except CmcException as ex:
            await context.bot.send_message(self.chat_id, str(ex))
        except BinanceException as ex:
            await context.bot.send_message(self.chat_id, str(ex))
