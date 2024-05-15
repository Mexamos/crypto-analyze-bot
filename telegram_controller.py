import math
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import List

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from database.client import DatabaseClient
from coinmarketcap_client import CoinmarketcapClient, CmcException
from binance_client import BinanceClient, BinanceException
from chart import ChartController, DataForChartNotFound, DataForIncomesTableNotFound
from config import Config


class RequestCurrenciesAction(Enum):
    BUY = 1
    SELL = 2
    ADD_DATA = 3
    SKIP = 4


class TelegramController:

    def __init__(
        self, db_client: DatabaseClient, cmc_client: CoinmarketcapClient, binance_cleint: BinanceClient,
        chart_controller: ChartController, config: Config, token: str, chat_id: str
    ) -> None:
        self.db_client = db_client
        self.cmc_client = cmc_client
        self.binance_cleint = binance_cleint
        self.chart_controller = chart_controller
        self.config = config
        self.app = Application.builder().token(token).build()
        self.chat_id = int(chat_id)

        self.app.add_handler(CommandHandler("prices_on_chart", self.prices_on_chart))
        self.app.add_handler(CommandHandler("incomes_table", self.incomes_table))
        self.app.add_handler(CommandHandler("list_current_currencies", self.list_current_currencies))
        self.app.add_handler(CommandHandler("list_income_currencies", self.list_income_currencies))

        job_queue = self.app.job_queue
        job_queue.run_repeating(self.request_currencies, interval=self.config.request_currencies_interval)

    def run_bot(self):
        # Run the bot until the user presses Ctrl-C
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)

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

        currencies = self.db_client.find_currency_price_symbols()
        if len(currencies) > 0:
            await update.message.reply_text('\n'.join(currencies))
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
                toAsset=currency['symbol'], fromAsset='USDT'
            )
            if isinstance(exchange_infos, list) and len(exchange_infos) > 0:
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
            date_time=datetime.now(timezone.utc),
        )

    async def caclculate_income_value(self, currency: dict) -> Decimal:
        currency_prices = self.db_client.find_currency_price_by_symbol(currency['symbol'])
        currency_price = currency_prices[0]

        old_price = currency_price.price
        new_price = Decimal(str(currency['quote']['USD']['price']))

        return (new_price * self.config.transactions_amount) - (old_price * self.config.transactions_amount)

    async def _create_income(self, currency: dict):
        value = await self.caclculate_income_value(currency)
        self.db_client.create_income(
            symbol=currency['symbol'],
            value=value,
            date_time=datetime.now(timezone.utc),
        )

    async def _sell_currency(self, context: ContextTypes.DEFAULT_TYPE, currency: dict):
        symbol = currency['symbol']
        try:
            # TODO add convert to USDT request

            # Send result prices chart
            chart_file = self.chart_controller.generate_chart_image(symbol)
            await context.bot.send_photo(self.chat_id, chart_file)

            # Add new income data
            await self._create_income(currency)

            # Delete existing currecny prices
            self.db_client.delete_currency_prices(symbol)
        except DataForChartNotFound as ex:
            await context.bot.send_message(self.chat_id, str(ex))

    async def request_currencies(self, context: ContextTypes.DEFAULT_TYPE):
        try:
            currencies = self.cmc_client.actual_trending_latest_currencies()
            currencies = await self._filter_currencies_by_binance(currencies)

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
