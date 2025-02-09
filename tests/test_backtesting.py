import os
import sys
import random
from datetime import datetime, timedelta
from multiprocessing import Pool

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from app.crypto.binance_client import BinanceClient


load_dotenv()

TOKEN = os.getenv('BOT_TOKEN')
BOT_CHAT_ID = os.getenv('BOT_CHAT_ID')

BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

SYMBOL = 'ETHUSDT'
INTERVAL = '1s'
HISTORICAL_PERIOD_IN_DAYS = 30
NUMBER_MISSING_SECONDS_BETWEEEN_SIGNALS = 30
INITIAL_TRADE_AMOUNT = 100

PROCESS_NUMBER_TO_LOAD_DATA = 5
CSV_FILE = f'{SYMBOL.lower()}_{INTERVAL}_binance_klines.csv'
DATA_HEADERS = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore']

# BTC
# WINDOW_HIGH = 20
# WINDOW_MEDIUM = 5
# WINDOW_LOW = 5
# WINDOW_VOLUME = 100
# 37, 10, 5, 13 - 90
# 20, 5, 5, 100 - 89
# 57, 21, 8, 17 - 93
# 24, 15, 5, 41 - 54
# 13, 6, 17, 94 - 46
# 20, 8, 8, 48 - 57
# 12, 20, 6, 100 - 109

# ETH
WINDOW_HIGH = 20
WINDOW_MEDIUM = 5
WINDOW_LOW = 5
WINDOW_VOLUME = 50
# 25, 20, 10, 56 - 89
# 18, 6, 11, 93 - 52
# 14, 15, 12, 85 - 23
# 14, 12, 11, 21 - 25
# 11, 21, 6, 93 - 34
# 27, 31, 5, 83 - 114

# SOL
# WINDOW_HIGH = 30
# WINDOW_MEDIUM = 5
# WINDOW_LOW = 5
# WINDOW_VOLUME = 30
# 67, 34, 7, 33 - 84
# 20, 13, 10, 26 - 37
# 44, 31, 18, 93 - 39
# 52, 29, 5, 69 - 59
# 30 10 6 48 - 105
# 30 5 5 30 - 119
# 59, 15, 13, 31 -143
# 32, 25, 10, 93 - 93
# 43, 36, 7, 11 - 55
# 25, 19, 10, 54 - 47



binance_client = BinanceClient(BINANCE_API_KEY, BINANCE_SECRET_KEY)


def remove_file(file_name):
    if os.path.exists(file_name):
        os.remove(file_name)


def data_type_conversion_to_file(df: pd.DataFrame):
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')
    df['Open'] = df['Open'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Close'] = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    return df


def calculate_loading_progress(
    start: datetime, current: datetime, finish: datetime,
) -> int:
    total_duration = current - start
    elapsed_duration = finish - start
    percentage_elapsed = (total_duration.total_seconds() / elapsed_duration.total_seconds()) * 100

    return min(round(percentage_elapsed), 100)


# Функция загрузки данных за заданный подинтервал
def load_data_for_subperiod(args):
    binance_client, start_date, end_date, output_file = args
    
    # Для отслеживания прогресса в данном подинтервале
    current_date = end_date
    
    # Открываем файл на запись (без заголовка, так как мы потом будем мерджить файлы)
    while current_date > start_date:
        # Запрос данных с параметром end_time = current_date
        klines = binance_client.kline_data(
            symbol=SYMBOL,
            interval=INTERVAL,
            end_time=current_date
        )
        if not klines:
            break

        df = pd.DataFrame(klines, columns=DATA_HEADERS)
        df = data_type_conversion_to_file(df)
        
        # Дописываем данные в файл
        df.to_csv(output_file, mode='a', index=False, header=False)
        
        # Для следующей итерации берем самую раннюю дату из полученных данных
        current_date = df.iloc[0]['Close Time'].to_pydatetime()
        
        # Отображение прогресса для данного подинтервала
        progress = ( (end_date - current_date).total_seconds() / (end_date - start_date).total_seconds() ) * 100
        sys.stdout.write(
            f"\rSegment {start_date.strftime('%Y-%m-%d %H:%M:%S')} - {end_date.strftime('%Y-%m-%d %H:%M:%S')}: {min(round(progress), 100)}%"
        )
        sys.stdout.flush()
    
    return output_file


# Функция объединения CSV файлов
def merge_csv_files(file_list, merged_file):
    merged_df = pd.DataFrame()
    for file in file_list:
        if os.path.exists(file):
            df = pd.read_csv(file, header=None, names=DATA_HEADERS)
            merged_df = pd.concat([merged_df, df])
            remove_file(file)
        else:
            print(f"Файл {file} не найден!")

    # Сортируем по дате открытия
    merged_df['Open Time'] = pd.to_datetime(merged_df['Open Time'])
    merged_df.sort_values('Open Time', inplace=True)
    merged_df.to_csv(merged_file, index=False, header=None)


def load_and_save_data_to_file():
    remove_file(CSV_FILE)

    end_date = datetime(2024, 12, 31)
    start_date = end_date - timedelta(days=HISTORICAL_PERIOD_IN_DAYS)
    segment_duration = (end_date - start_date) / PROCESS_NUMBER_TO_LOAD_DATA

    segments = []
    file_list = []
    for i in range(PROCESS_NUMBER_TO_LOAD_DATA):
        seg_start = start_date + i * segment_duration
        seg_end = start_date + (i + 1) * segment_duration
        output_file = f'{SYMBOL.lower()}_{INTERVAL}_binance_klines_segment_{i}.csv'
        remove_file(output_file)

        segments.append((binance_client, seg_start, seg_end, output_file))
        file_list.append(output_file)

    # Используем пул процессов для параллельной загрузки
    with Pool(processes=PROCESS_NUMBER_TO_LOAD_DATA) as pool:
        pool.map(load_data_for_subperiod, segments)

    # Мержим полученные файлы
    merge_csv_files(file_list, CSV_FILE)


# if __name__ == '__main__':
#     start_time = datetime.now()
#     load_and_save_data_to_file()
#     print('spent time', datetime.now() - start_time)



def data_type_conversion_from_file(df: pd.DataFrame):
    df['Open Time'] = pd.to_datetime(df['Open Time'])
    df['Close Time'] = pd.to_datetime(df['Close Time'])
    df['Open'] = df['Open'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Close'] = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    return df


def add_simple_moving_average(
    df: pd.DataFrame, window_high: int,
    window_medium: int, window_low: int, window_volume
):
    # Старший таймфрейм
    df["sma_high"] = df["Close"].rolling(window=window_high).mean()
    # Средний таймфрейм
    df["sma_medium"] = df["Close"].rolling(window=window_medium).mean()
    # Младший таймфрейм
    df["sma_low"] = df["Close"].rolling(window=window_low).mean()

    df["vol_ma"] = df["Volume"].rolling(window=window_volume).mean()

    return df


def calculate_signals(df):
    """
        Определение сигналов покупки и продажи на основе нескольких таймфреймов с учетом объемов
    """
    position = None  # Текущее состояние: None (нет позиции) или "long"
    n = len(df)
    buy_signals = [False] * n
    sell_signals = [False] * n

    i = 0
    while i < n:
        # Получаем строку (данные для i-й свечи)
        row = df.iloc[i]
        if position is None:
            # Условие для покупки: цена закрытия выше всех SMA и объём больше vol_ma
            if (
                row["Close"] > row["sma_high"] and
                row["Close"] > row["sma_medium"] and
                row["Close"] > row["sma_low"] and
                row["Volume"] > row["vol_ma"]
            ):
                buy_signals[i] = True
                position = "long"
                # Пропускаем следующие 30 строк (30 секунд)
                i += NUMBER_MISSING_SECONDS_BETWEEEN_SIGNALS
                continue  # Переход к следующей итерации while
        else:  # position == "long"
            # Условие для продажи: цена закрытия ниже всех SMA и объём больше vol_ma
            if (
                row["Close"] < row["sma_high"] and
                row["Close"] < row["sma_medium"] and
                row["Close"] < row["sma_low"] and
                row["Volume"] > row["vol_ma"]
            ):
                sell_signals[i] = True
                position = None
                # Пропускаем следующие 30 строк (30 секунд)
                i += NUMBER_MISSING_SECONDS_BETWEEEN_SIGNALS
                continue
        i += 1

    df["buy_signal"] = buy_signals
    df["sell_signal"] = sell_signals
    return df


def calculate_profit(df, initial_balance):
    balance = initial_balance
    position = 0  # Количество купленной криптовалюты
    for index, row in df.iterrows():
        if row["buy_signal"] and balance > 0:
            position = balance / row["Close"]  # Покупка криптовалюты
            balance = 0
        elif row["sell_signal"] and position > 0:
            balance = position * row["Close"]  # Продажа криптовалюты
            position = 0
    # Финальная стоимость (если позиция не закрыта, учитываем её текущую стоимость)
    final_value = balance + (position * df.iloc[-1]["Close"] if position > 0 else 0)
    return final_value - initial_balance


def show_data_on_plot(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df["Open Time"], df["Close"], label="Close Price", linewidth=1.5)
    # Добавление точек покупки и продажи
    plt.scatter(df.loc[df["buy_signal"], "Open Time"], df.loc[df["buy_signal"], "Close"], label="Buy Signal", color="green", marker="^", alpha=1)
    plt.scatter(df.loc[df["sell_signal"], "Open Time"], df.loc[df["sell_signal"], "Close"], label="Sell Signal", color="red", marker="v", alpha=1)

    plt.title("Buy and Sell Signals Based on Multi-Timeframe Trend Analysis with Volume")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()


def optimize_params(param_set):
    temp_df, initial_trade_amount, w_high, w_medium, w_low, w_volume = param_set
    temp_df = add_simple_moving_average(temp_df, w_high, w_medium, w_low, w_volume)
    temp_df = calculate_signals(temp_df)
    profit = calculate_profit(temp_df, initial_trade_amount)
    return (w_high, w_medium, w_low, w_volume, profit)


def apply_trading_algorithm_to_historical_data():
    df = pd.read_csv(CSV_FILE, header=None, names=DATA_HEADERS)
    df = data_type_conversion_from_file(df)

    # df = add_simple_moving_average(df, WINDOW_HIGH, WINDOW_MEDIUM, WINDOW_LOW, WINDOW_VOLUME)
    # df = calculate_signals(df)
    # profit = calculate_profit(df, INITIAL_TRADE_AMOUNT)
    # print(f"Final Profit: {profit:.2f} USDT")


    num_samples = 50
    random_params = [
        (
            df.copy(),
            INITIAL_TRADE_AMOUNT,
            random.randint(10, 100), 
            random.randint(5, 50), 
            random.randint(5, 20), 
            random.randint(10, 100)
        ) 
        for _ in range(num_samples)
    ]
    with Pool(processes=8) as pool:
        results = pool.map(optimize_params, random_params)
    best_params = max(results, key=lambda x: x[4])
    print('best_params', best_params)


if __name__ == '__main__':
    start_time = datetime.now()
    apply_trading_algorithm_to_historical_data()
    print('spent time', datetime.now() - start_time)

