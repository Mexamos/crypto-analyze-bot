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

BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

SYMBOL = 'SOLUSDT'
INTERVAL = '15m'
HISTORICAL_PERIOD_IN_DAYS = 30
NUMBER_MISSING_SECONDS_BETWEEEN_SIGNALS = 1
MIN_SECONDS_BETWEEN_TRADES = 60  # enforce minimum 60 sec gap between trades to reduce overtrading
INITIAL_TRADE_AMOUNT = 300

PROCESS_NUMBER_TO_LOAD_DATA = 5
CSV_FILE = f'{SYMBOL.lower()}_{INTERVAL}_binance_klines.csv'
DATA_HEADERS = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore']

# SOL
WINDOW_HIGH = 200
WINDOW_MEDIUM = 120
WINDOW_LOW = 80
WINDOW_VOLUME = 30

# Минимальное значение ADX для входа в сделку
ADX_THRESHOLD = 25
# Период для расчёта ATR
ATR_PERIOD = 15
# Множитель для стоп-лосса
ATR_STOP_MULTIPLIER = 1.5

# Risk & Execution Parameters
RISK_PCT = 0.02          # Risk 2% of account per trade
COMMISSION_RATE = 0.001  # 0.1% per trade commission
SLIPPAGE_PCT = 0.001     # Assume 0.1% slippage on both entry and exit

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

    end_date = datetime(2024, 11, 1)
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


def add_atr(df: pd.DataFrame, period):
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['atr'] = df['TR'].rolling(window=period).mean()
    return df


def add_adx(df: pd.DataFrame, period):
    # Расчёт Directional Movement
    df['up_move'] = df['High'] - df['High'].shift(1)
    df['down_move'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['-DM'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    # Истинный диапазон (True Range)
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    # Сглаживание по методу Wilder'a
    df['TR_sum'] = df['TR'].rolling(window=period).sum()
    df['+DM_sum'] = df['+DM'].rolling(window=period).sum()
    df['-DM_sum'] = df['-DM'].rolling(window=period).sum()
    
    df['+DI'] = 100 * (df['+DM_sum'] / df['TR_sum'])
    df['-DI'] = 100 * (df['-DM_sum'] / df['TR_sum'])
    df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df['adx'] = df['DX'].rolling(window=period).mean()
    return df


def calculate_signals(df, adx_threshold):
    """
    Определяет сигналы покупки и продажи с учетом:
      - Мульти-SMA (разные таймфреймы)
      - Фильтра по ADX (торгуем только при силном тренде)
      - ATR-стоп (выход, если цена падает ниже уровня стоп-лосса)
    """
    position = None     # Текущая позиция: None или "long"
    entry_price = None  # Цена входа в позицию
    stop_loss = None    # Уровень стоп-лосса (рассчитывается по ATR)
    n = len(df)
    buy_signals = [False] * n
    sell_signals = [False] * n

    i = 0
    while i < n:
        row = df.iloc[i]
        # Если ADX ниже порога, пропускаем сигнал (если нет открытой позиции)
        if pd.notna(row["adx"]) and row["adx"] < adx_threshold:
            # Если позиция открыта, всё равно проверяем стоп-лосс
            if position is not None and stop_loss is not None and row["Low"] < stop_loss:
                sell_signals[i] = True
                position = None
                entry_price = None
                stop_loss = None
                i += NUMBER_MISSING_SECONDS_BETWEEEN_SIGNALS
                continue
            i += 1
            continue

        if position is None:
            # Условия для входа в позицию (покупка):
            if (row["Close"] > row["sma_high"] and
                row["Close"] > row["sma_medium"] and
                row["Close"] > row["sma_low"] and
                row["Volume"] > row["vol_ma"]):
                buy_signals[i] = True
                position = "long"
                entry_price = row["Close"]
                # Рассчитываем уровень стоп-лосса по ATR
                if pd.notna(row["atr"]):
                    stop_loss = entry_price - ATR_STOP_MULTIPLIER * row["atr"]
                else:
                    stop_loss = entry_price * 0.98
                i += NUMBER_MISSING_SECONDS_BETWEEEN_SIGNALS
                continue
        else:
            if (
                stop_loss is not None and row["Low"] < stop_loss
            ) or (
                row["Close"] < row["sma_high"] and
                row["Close"] < row["sma_medium"] and
                row["Close"] < row["sma_low"] and
                row["Volume"] > row["vol_ma"]
            ):
                sell_signals[i] = True
                position = None
                entry_price = None
                stop_loss = None
                i += NUMBER_MISSING_SECONDS_BETWEEEN_SIGNALS
                continue

        i += 1

    df["buy_signal"] = buy_signals
    df["sell_signal"] = sell_signals
    return df


def calculate_profit(df, initial_balance, commission_rate=0.001):
    balance = initial_balance
    position = 0  # Количество купленной криптовалюты
    for index, row in df.iterrows():
        if row["buy_signal"] and balance > 0:
            # Покупаем крипту с учетом комиссии: комиссия уменьшает купленное количество
            position = (balance / row["Close"]) * (1 - commission_rate)
            balance = 0
        elif row["sell_signal"] and position > 0:
            # Продаем крипту с учетом комиссии: комиссия уменьшает получаемую сумму
            balance = position * row["Close"] * (1 - commission_rate)
            position = 0
    # Если позиция осталась открытой, учитываем ее текущую стоимость без комиссии
    final_value = balance + (position * df.iloc[-1]["Close"] if position > 0 else 0)
    return final_value - initial_balance


# New simulation function with risk management, slippage, fee adjustment, and trade frequency control
def simulate_trades(df, initial_balance, risk_pct=RISK_PCT, commission_rate=COMMISSION_RATE, slippage_pct=SLIPPAGE_PCT, min_seconds_between_trades=MIN_SECONDS_BETWEEN_TRADES):
    balance = initial_balance
    position = 0      # Number of coins held
    entry_price = None
    stop_loss = None
    entry_cost = 0
    last_trade_time = None
    trades = []
    
    # Ensure the DataFrame is sorted by time
    df = df.sort_values('Open Time').reset_index(drop=True)
    
    for index, row in df.iterrows():
        current_time = row["Open Time"]
        current_price = row["Close"]
        
        # Enforce a minimum gap between trades
        if last_trade_time is not None:
            time_diff = (current_time - last_trade_time).total_seconds()
        else:
            time_diff = np.inf
        
        # No open position: check for buy signal and time gap
        if position == 0:
            if row["buy_signal"] and time_diff >= min_seconds_between_trades:
                entry_price = current_price
                # Set stop-loss based on ATR (or fallback to a 2% drop)
                if pd.notna(row["atr"]) and row["atr"] > 0:
                    stop_loss = entry_price - ATR_STOP_MULTIPLIER * row["atr"]
                else:
                    stop_loss = entry_price * 0.98
                risk_per_coin = entry_price - stop_loss
                if risk_per_coin <= 0:
                    continue
                risk_amount = balance * risk_pct
                position = risk_amount / risk_per_coin
                effective_buy_price = current_price * (1 + slippage_pct)
                entry_cost = position * effective_buy_price * (1 + commission_rate)
                if entry_cost > balance:
                    position = balance / (effective_buy_price * (1 + commission_rate))
                    entry_cost = balance
                balance -= entry_cost
                trade_entry_time = current_time
        else:
            # If position is open, check for exit conditions: stop-loss hit or sell signal
            exit_trade = False
            exit_price = None
            if row["Low"] < stop_loss:
                exit_trade = True
                exit_price = stop_loss * (1 - slippage_pct)
            elif row["sell_signal"] and time_diff >= min_seconds_between_trades:
                exit_trade = True
                exit_price = current_price * (1 - slippage_pct)
            
            if exit_trade:
                proceeds = position * exit_price * (1 - commission_rate)
                balance += proceeds
                trade_profit = proceeds - entry_cost
                trades.append({
                    'entry_time': trade_entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit': trade_profit
                })
                # Reset trade variables and update last trade time
                position = 0
                entry_price = None
                stop_loss = None
                entry_cost = 0
                last_trade_time = current_time
    
    # If a position remains open, close it at the final available price
    if position > 0:
        final_price = df.iloc[-1]["Close"] * (1 - slippage_pct)
        proceeds = position * final_price * (1 - commission_rate)
        balance += proceeds
        trade_profit = proceeds - entry_cost
        trades.append({
            'entry_time': trade_entry_time,
            'exit_time': df.iloc[-1]["Open Time"],
            'entry_price': entry_price,
            'exit_price': final_price,
            'profit': trade_profit
        })
        position = 0
    
    final_equity = balance
    total_profit = final_equity - initial_balance
    return total_profit, trades


def compute_performance_metrics(trades, initial_balance):
    if not trades:
        return {'profit': 0, 'final_equity': initial_balance, 'max_drawdown': 0,
                'profit_factor': np.nan, 'win_rate': np.nan, 'sharpe': np.nan,
                'equity_curve': [initial_balance]}
    
    equity_curve = [initial_balance]
    for trade in trades:
        equity_curve.append(equity_curve[-1] + trade['profit'])
    
    # Maximum Drawdown calculation
    peak = -np.inf
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd
    
    # Profit Factor and Win Rate
    wins = [t['profit'] for t in trades if t['profit'] > 0]
    losses = [t['profit'] for t in trades if t['profit'] < 0]
    profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else np.nan
    win_rate = len(wins) / len(trades) if trades else np.nan
    
    # Sharpe Ratio based on trade returns (normalized by initial balance)
    trade_returns = [t['profit'] / initial_balance for t in trades]
    if np.std(trade_returns) > 0:
        sharpe = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(len(trade_returns))
    else:
        sharpe = np.nan
    
    return {
        'profit': equity_curve[-1] - initial_balance,
        'final_equity': equity_curve[-1],
        'max_drawdown': max_dd,
        'profit_factor': profit_factor,
        'win_rate': win_rate,
        'sharpe': sharpe,
        'equity_curve': equity_curve
    }


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
    temp_df = add_adx(temp_df, ATR_PERIOD)
    temp_df = add_atr(temp_df, ATR_PERIOD)
    temp_df = calculate_signals(temp_df, ADX_THRESHOLD)
    profit = calculate_profit(temp_df, initial_trade_amount)

    # print('["buy_signal"]', temp_df["buy_signal"].sum())
    # print('["sell_signal"]', temp_df["sell_signal"].sum())
    return (w_high, w_medium, w_low, w_volume, profit)


def apply_trading_algorithm_to_historical_data():
    df = pd.read_csv(CSV_FILE, header=None, names=DATA_HEADERS)
    df = data_type_conversion_from_file(df)

    # df = add_simple_moving_average(df, WINDOW_HIGH, WINDOW_MEDIUM, WINDOW_LOW, WINDOW_VOLUME)
    # df = add_adx(df, ATR_PERIOD)
    # df = add_atr(df, ATR_PERIOD)
    # df = calculate_signals(df, ADX_THRESHOLD)
    # # profit = calculate_profit(df, INITIAL_TRADE_AMOUNT)
    # # print(f"Final Profit: {profit:.2f} USDT")

    # profit, trades = simulate_trades(df, INITIAL_TRADE_AMOUNT)
    # metrics = compute_performance_metrics(trades, INITIAL_TRADE_AMOUNT)
    # print(f"Final Profit: {profit:.2f} USDT")
    # print(f"Final Equity: {metrics['final_equity']:.2f} USDT")
    # print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    # print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    # print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
    # print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")


    num_samples = 150
    random_params = [
        (
            df.copy(),
            INITIAL_TRADE_AMOUNT,
            random.randint(100, 5000), 
            random.randint(50, 3000), 
            random.randint(5, 1000), 
            random.randint(30, 5000)
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

