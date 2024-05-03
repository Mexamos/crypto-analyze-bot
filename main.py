import os

import requests
import random
import hmac
import hashlib
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv('BOT_TOKEN')

import logging

from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from models import Base, Currency
from sqlalchemy import create_engine

from sqlalchemy.orm import Session
from sqlalchemy import select

engine = create_engine("sqlite://", echo=True)

Base.metadata.create_all(engine)






# Структура данных для реального бота
# список валют что мы уже купили
# название валюты
# цена за которую купили (~ratio)
# время когда купили
# orderId


# Структура данных для бота анализатора - смотри что возвращают CMC ендпоинты
# symbol
# quote.USD.price
# quote.USD.percent_change_24h
# текущий timestamp


# Trending Gainers & Losers
# https://coinmarketcap.com/api/documentation/v1/#operation/getV1CryptocurrencyTrendingGainerslosers

# Trending Latest
# https://coinmarketcap.com/api/documentation/v1/#operation/getV1CryptocurrencyTrendingLatest

# from requests import Request, Session
# from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
# import json

# url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/trending/latest'
# parameters = {
#   'start':'1',
#   'limit':'100',
#   'time_period': '24h',
# }
# headers = {
#   'Accepts': 'application/json',
#   'X-CMC_PRO_API_KEY': '2dd1953d-d7cc-4e67-8424-1bc888e49ab1',
# }

# session = Session()
# session.headers.update(headers)

# try:
#     response = session.get(url, params=parameters)
#     data = response.json()
# except (ConnectionError, Timeout, TooManyRedirects) as e:
#     print(e)


# print('len 1', len(data['data']))


# def filter_data(currency):
#     if currency['quote']['USD']['percent_change_24h'] > 0:
#         return True 
#     else:
#         return False

# filtered_data = filter(filter_data, data['data'])

# def sort_data(currency):
#     return currency['quote']['USD']['percent_change_24h']


# sorted_data = sorted(filtered_data, key=sort_data, reverse=True)

# print('len 2', len(sorted_data))
# print('sorted_data[0]', sorted_data[0])
# print('sorted_data[last]', sorted_data[len(sorted_data) - 1])





















# BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
# BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

# https://binance-docs.github.io/apidocs/spot/en/#convert-endpoints

# Для аналитики использовать один из этих двух
# GET /sapi/v1/convert/exchangeInfo
# POST /sapi/v1/convert/getQuote

# POST /sapi/v1/convert/getQuote
# Send quote request
# Return quoteId

# POST /sapi/v1/convert/acceptQuote
# Accept Quote (TRADE)
# Return orderId

# GET /sapi/v1/convert/orderStatus
# Order status (USER_DATA)
# Check order status


# # GET /sapi/v1/convert/exchangeInfo
# # List All Convert Pairs
# url = 'https://api.binance.com/sapi/v1/convert/exchangeInfo?fromAsset=USDT&toAsset=SOL'
# headers = {
#     'Content-Type': 'application/json',
#     'X-MBX-APIKEY': BINANCE_API_KEY
# }
# response = requests.get(url, headers=headers)
# print('response', response.status_code, response.json())


# # query_string = 'fromAsset=USDT&toAsset=SOL&fromAmount=5'
# url = 'https://api.binance.com/sapi/v1/convert/getQuote?fromAsset=USDT&toAsset=SOL&fromAmount=5&timestamp=1712336207023&signature=9ee524e170fbf8fb288fee801da4862b457a293ad6ef70f4a31a6e8c39cee3a4'
# headers = {
#     'Content-Type': 'application/json',
#     'X-MBX-APIKEY': BINANCE_API_KEY
# }

# # m = hmac.new(BINANCE_SECRET_KEY.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256)
# # signature = m.hexdigest()
# # print('signature', signature)

# response = requests.post(url, headers=headers)
# print('response', response.status_code, response.json())









from io import BytesIO

#  TELEGRAM !!!!!!!!
# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Help!")


async def add_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with Session(engine) as session:
        currency = Currency(
            symbol='SOL',
            price=str(random.randint(1, 100)),
            percent_change_24h=str(random.randint(1, 10)),
            timestamp=datetime.now(timezone.utc),
        )
        session.add(currency)
        session.commit()


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    # await update.message.reply_text(update.message.text)


    connection = engine.connect()
    query = select(Currency)

    plot_file = BytesIO()

    # np.random.seed(123456)
    df = pd.read_sql(query, connection)
    print('df', df, type(df))

    ts = df[['timestamp', 'price']]

    print('ts', ts, type(ts))

    plot = ts.plot()
    fig = plot.get_figure()
    fig.savefig(plot_file, format='png')
    plot_file.seek(0)

    await update.message.reply_photo(plot_file)


    # with Session(engine) as session:
    #     curencies = session.scalars(select(Currency))
    #     print('curencies', list(curencies))



def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("add", add_data))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
