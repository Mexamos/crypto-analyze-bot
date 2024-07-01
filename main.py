import os

from dotenv import load_dotenv

from app.bot_controller import BotController
from app.database.client import DatabaseClient
from app.crypto.coinmarketcap_client import CoinmarketcapClient
from app.crypto.binance_client import BinanceClient
from app.analytics.chart import ChartController
from app.config import Config
from app.analytics.google_sheets_client import GoogleSheetsClient
from app.monitoring.sentry import SentryClient

load_dotenv()

TOKEN = os.getenv('BOT_TOKEN')
BOT_CHAT_ID = os.getenv('BOT_CHAT_ID')

COIN_MARKET_CAP_API_KEY = os.getenv('COIN_MARKET_CAP_API_KEY')

BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

CREDENTIALS_FILE_PATH = os.getenv('CREDENTIALS_FILE_PATH')
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')

SENTRY_DSN = os.getenv('SENTRY_DSN')

# TODO Add for requests raises exceptions !!!!!!!!!!

# TODO urls
# /v2/cryptocurrency/quotes/historical
# /v1/cryptocurrency/listings/historical

# TODO порешать ошибки из сентри

# TODO написать Readme.md

# TODO написать тесты

# TODO добавить линтер(-ы)

def main():
    config = Config()
    db_client = DatabaseClient()

    chart_controller = ChartController(db_client, config)
    google_sheets_client = GoogleSheetsClient(CREDENTIALS_FILE_PATH, SPREADSHEET_ID)

    cmc_client = CoinmarketcapClient(COIN_MARKET_CAP_API_KEY, config)
    binance_cleint = BinanceClient(BINANCE_API_KEY, BINANCE_SECRET_KEY)

    sentry_client = SentryClient(SENTRY_DSN, config)

    telegram_controller = BotController(
        db_client, cmc_client, binance_cleint,
        chart_controller, google_sheets_client,
        sentry_client, config, TOKEN, BOT_CHAT_ID
    )
    telegram_controller.init_bot()
    # telegram_controller.restore_unsold_currencies()
    telegram_controller.run_bot()


if __name__ == "__main__":
    main()


# import backtrader as bt
# import pandas as pd

# # Example historical data
# data = {
#     'Date': ['2023-06-01', '2023-06-02', '2023-06-03', '2023-06-04', '2023-06-05', '2023-06-06', '2023-06-07'],
#     'Open': [100, 105, 103, 108, 110, 107, 109],
#     'High': [105, 107, 106, 111, 113, 109, 112],
#     'Low': [95, 102, 100, 106, 108, 105, 107],
#     'Close': [102, 106, 104, 110, 111, 108, 110],
#     'Volume': [1000, 1100, 1050, 1200, 1150, 1300, 1250]
# }

# df = pd.DataFrame(data)
# df['Date'] = pd.to_datetime(df['Date'])
# df.set_index('Date', inplace=True)

# class MomentumStrategy(bt.Strategy):
#     params = (('window', 5), ('stop_loss_pct', 0.05), ('take_profit_pct', 0.10))

#     def __init__(self):
#         self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.window)
#         self.momentum = self.data.close - self.data.close(-self.params.window)
#         self.roc = (self.data.close - self.data.close(-self.params.window)) / self.data.close(-self.params.window) * 100
#         self.buy_price = None

#     def next(self):
#         if self.position:
#             current_price = self.data.close[0]
#             if current_price <= self.buy_price * (1 - self.params.stop_loss_pct):
#                 self.sell()
#                 self.buy_price = None
#             elif current_price >= self.buy_price * (1 + self.params.take_profit_pct):
#                 self.sell()
#                 self.buy_price = None
#         else:
#             if self.data.close[0] > self.sma[0] and self.momentum[0] > 1 and self.roc[0] > 1:
#                 self.buy()
#                 self.buy_price = self.data.close[0]
#             elif self.data.close[0] < self.sma[0] and self.momentum[0] < -1 and self.roc[0] < -1:
#                 self.sell()

# # Initialize Cerebro engine
# cerebro = bt.Cerebro()

# # Add strategy to Cerebro
# cerebro.addstrategy(MomentumStrategy)

# # Load data into backtrader
# data = bt.feeds.PandasData(dataname=df)
# cerebro.adddata(data)

# # Set initial capital and run backtest
# cerebro.broker.setcash(10000)
# print(f"Starting Portfolio Value: {cerebro.broker.getvalue()}")
# cerebro.run()
# print(f"Final Portfolio Value: {cerebro.broker.getvalue()}")

# # Plot the results
# cerebro.plot()

