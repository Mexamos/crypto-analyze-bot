import logging
from typing import Dict

from telegram.ext import ContextTypes
from telegram import Update

from app.crypto.coinmarketcap_client import CoinmarketcapClient
from app.crypto.binance_client import BinanceClient
from app.config import Config
from app.monitoring.sentry import SentryClient


class AnalyzeProcessor:

    def __init__(
        self,
        binance_client: BinanceClient,
        coinmarketcap_client: CoinmarketcapClient,
        sentry_client: SentryClient,
        config: Config,
        chat_ids: list,
    ):
        self.binance_client = binance_client
        self.coinmarketcap_client = coinmarketcap_client

        self.sentry_client = sentry_client
        self.config = config
        self.symbols = self.config.analyze_symbols
        self.symbol_name_dict = {
            symbol: name.lower() for symbol, name in zip(self.symbols, self.config.analyze_names)
        }
        self.change_percentages = {
            symbol: percentage for symbol, percentage in zip(self.symbols, self.config.analyze_change_percentages)
        }

        self.chat_ids = chat_ids
        self.logger = logging.getLogger(__name__)

        self.get_cmc_ids()

    def get_cmc_ids(self) -> Dict[str, int]:
        cmc_id_map = self.coinmarketcap_client.get_id_map(symbol=','.join(self.symbols))
        self.symbol_to_cmc_id = {}
        for coin in cmc_id_map['data']:
            if coin['slug'] == self.symbol_name_dict[coin['symbol']]:
                self.symbol_to_cmc_id[coin['symbol']] = coin['id']

    def _generate_price_go_up_message(self, symbol: str, price_info: dict) -> str:
        return (
            f"📈 ⬆️ 💹 Price go up for {symbol}:\n"
            f"Price: {price_info['price']} {self.config.currency_conversion}\n"
            f"24h Change: {price_info['percent_change_24h']}%\n"
            f"7d Change: {price_info['percent_change_7d']}%\n"
            f"30d Change: {price_info['percent_change_30d']}%\n"
            f"90d Change: {price_info['percent_change_90d']}%\n"
        )

    def _generate_price_go_down_message(self, symbol: str, price_info: dict) -> str:
        return (
            f"📉 ⬇️ 🔻 Price go down for {symbol}:\n"
            f"Price: {price_info['price']} {self.config.currency_conversion}\n"
            f"24h Change: {price_info['percent_change_24h']}%\n"
            f"7d Change: {price_info['percent_change_7d']}%\n"
            f"30d Change: {price_info['percent_change_30d']}%\n"
            f"90d Change: {price_info['percent_change_90d']}%\n"
        )

    def _generate_price_without_change_message(self, symbol: str, price_info: dict) -> str:
        return (
            f"ℹ️ No significant change for {symbol}:\n"
            f"Price: {price_info['price']} {self.config.currency_conversion}\n"
            f"24h Change: {price_info['percent_change_24h']}%\n"
            f"7d Change: {price_info['percent_change_7d']}%\n"
            f"30d Change: {price_info['percent_change_30d']}%\n"
            f"90d Change: {price_info['percent_change_90d']}%\n"
        )

    async def process(self, context: ContextTypes.DEFAULT_TYPE):
        try:
            account_balances = self.binance_client.get_account_balance(assets=self.symbols)

            messages = []
            for symbol in self.symbols:
                response = self.coinmarketcap_client.get_quotes_latest(
                    self.symbol_to_cmc_id[symbol], convert=self.config.currency_conversion
                )
                result = response.get('data', {}).get(str(self.symbol_to_cmc_id[symbol]), {})
                price_info = result['quote'][self.config.currency_conversion]

                if (
                    price_info['percent_change_24h'] >= self.change_percentages[symbol]
                    and price_info['percent_change_7d'] >= self.change_percentages[symbol]
                    and account_balances.get(symbol, 0) > 0
                ):
                    messages.append(self._generate_price_go_up_message(symbol, price_info))

                if (
                    price_info['percent_change_24h'] <= -self.change_percentages[symbol]
                    and price_info['percent_change_7d'] <= -self.change_percentages[symbol]
                ):
                    messages.append(self._generate_price_go_down_message(symbol, price_info))

            if messages:
                for chat_id in self.chat_ids:
                    message = "\n\n".join(messages)
                    await context.bot.send_message(chat_id, message)

        except Exception as ex:
            self.logger.info(f"Error processing: {ex}")
            self.sentry_client.capture_exception(ex)

    async def get_price_changes(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message.chat_id not in self.chat_ids:
            return

        try:
            messages = []
            for symbol in self.symbols:
                response = self.coinmarketcap_client.get_quotes_latest(
                    self.symbol_to_cmc_id[symbol], convert=self.config.currency_conversion
                )
                result = response.get('data', {}).get(str(self.symbol_to_cmc_id[symbol]), {})
                price_info = result['quote'][self.config.currency_conversion]

                if (
                    price_info['percent_change_24h'] >= self.change_percentages[symbol]
                    and price_info['percent_change_7d'] >= self.change_percentages[symbol]
                ):
                    messages.append(self._generate_price_go_up_message(symbol, price_info))

                elif (
                    price_info['percent_change_24h'] <= -self.change_percentages[symbol]
                    and price_info['percent_change_7d'] <= -self.change_percentages[symbol]
                ):
                    messages.append(self._generate_price_go_down_message(symbol, price_info))

                else:
                    messages.append(self._generate_price_without_change_message(symbol, price_info))

            if messages:
                for chat_id in self.chat_ids:
                    message = "\n\n".join(messages)
                    await context.bot.send_message(chat_id, message)

        except Exception as ex:
            self.logger.info(f"Error processing: {ex}")
            self.sentry_client.capture_exception(ex)
