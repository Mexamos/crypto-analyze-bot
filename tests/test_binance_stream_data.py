import time
import json
import hmac
import hashlib
import requests
import websocket  # pip install websocket-client
import pandas as pd
import numpy as np
from datetime import datetime

# ===============================
# Настройки API и торговых параметров
# ===============================
API_KEY = "ВАШ_API_KEY"
SECRET_KEY = "ВАШ_SECRET_KEY"

BASE_URL = "https://api.binance.com"

# Торговая пара и базовые настройки
SYMBOL = "ETHUSDT"
BASE_ASSET = SYMBOL.replace("USDT", "")  # например, ETH

# Настройки индикаторов (количество свечей для скользящих средних)
WINDOW_HIGH = 50    # для sma_high
WINDOW_MEDIUM = 25  # для sma_medium
WINDOW_LOW = 10     # для sma_low
VOL_WINDOW = 10     # для объёмной скользящей средней

# Минимальные балансы для торговли (укажите реальные значения)
MIN_USDT_BALANCE = 10      # минимальный баланс USDT для покупки
MIN_ASSET_AMOUNT = 0.001   # минимальное количество базового актива для продажи

# ===============================
# Глобальные переменные
# ===============================
global_df = pd.DataFrame(
    columns=["Open Time", "Open", "High", "Low", "Close", "Volume", "Close Time"]
)
global_position = None  # None, если позиции нет; "long" – если куплено

# ===============================
# Функции для подписи и отправки REST-запросов
# ===============================
def sign_params(params: dict) -> dict:
    """
    Добавляет timestamp и signature к параметрам запроса.
    """
    params["timestamp"] = int(time.time() * 1000)
    query_string = "&".join([f"{key}={params[key]}" for key in sorted(params)])
    signature = hmac.new(SECRET_KEY.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256).hexdigest()
    params["signature"] = signature
    return params

def send_signed_request(http_method: str, url_path: str, params: dict):
    """
    Отправляет подписанный запрос к REST API Binance.
    """
    params = sign_params(params)
    url = BASE_URL + url_path
    headers = {"X-MBX-APIKEY": API_KEY}
    
    try:
        if http_method == "GET":
            response = requests.get(url, params=params, headers=headers)
        elif http_method == "POST":
            response = requests.post(url, params=params, headers=headers)
        else:
            raise ValueError("Неподдерживаемый метод: " + http_method)
        data = response.json()
        if response.status_code != 200:
            print(f"Ошибка {response.status_code}: {data}")
        return data
    except Exception as e:
        print("Ошибка при выполнении запроса:", e)
        return None

def create_order(symbol: str, side: str, order_type: str, quantity: float):
    """
    Создаёт ордер (рыночный) через REST API Binance.
    """
    params = {
        "symbol": symbol,
        "side": side,
        "type": order_type,
        "quantity": quantity
    }
    return send_signed_request("POST", "/api/v3/order", params)

def get_account_info():
    """
    Получает информацию об аккаунте, включая балансы.
    """
    return send_signed_request("GET", "/api/v3/account", {})

def get_asset_balance(asset: str) -> float:
    """
    Возвращает доступный (free) баланс указанного актива.
    """
    account_info = get_account_info()
    if account_info is None or "balances" not in account_info:
        print("Не удалось получить информацию об аккаунте.")
        return 0.0
    for balance in account_info["balances"]:
        if balance["asset"] == asset:
            return float(balance["free"])
    return 0.0

# ===============================
# Функции для расчёта сигналов и исполнения сделок
# ===============================
def update_signals_and_trade():
    """
    Рассчитывает скользящие средние и, если условия выполнены, отправляет ордера.
    """
    global global_df, global_position

    # Для расчёта индикаторов нужно минимум WINDOW_HIGH свечей
    if len(global_df) < WINDOW_HIGH:
        return

    df = global_df.copy()

    # Вычисление скользящих средних для цены закрытия
    df["sma_high"] = df["Close"].rolling(window=WINDOW_HIGH).mean()
    df["sma_medium"] = df["Close"].rolling(window=WINDOW_MEDIUM).mean()
    df["sma_low"] = df["Close"].rolling(window=WINDOW_LOW).mean()
    # Скользящая средняя для объёма
    df["vol_ma"] = df["Volume"].rolling(window=VOL_WINDOW).mean()

    # Берём самую последнюю свечу
    latest = df.iloc[-1]

    # --- Логика сигналов ---
    # Сигнал на покупку: если позиции нет и цена закрытия выше всех SMA, а объём выше средней величины объёма
    if global_position is None:
        if (latest["Close"] > latest["sma_high"] and
            latest["Close"] > latest["sma_medium"] and
            latest["Close"] > latest["sma_low"] and
            latest["Volume"] > latest["vol_ma"]):

            # Получаем баланс USDT
            available_usdt = get_asset_balance("USDT")
            if available_usdt < MIN_USDT_BALANCE:
                print(f"[{datetime.now()}] Недостаточно средств для покупки: {available_usdt} USDT")
                return

            # Рассчитываем количество базового актива для покупки
            quantity = available_usdt / latest["Close"]
            quantity = round(quantity, 6)  # проверьте точность (stepSize) для SYMBOL
            print(f"[{datetime.now()}] Сигнал BUY: Цена {latest['Close']}, Покупаем {quantity} {BASE_ASSET}")

            order = create_order(SYMBOL, "BUY", "MARKET", quantity)
            print("Ответ ордера на покупку:", order)
            if order and "status" in order and order["status"] == "FILLED":
                global_position = "long"
            else:
                # Если ордер не выполнен, можно установить обработку ошибок
                global_position = "long"  # или оставить позицию не открытой
    # Сигнал на продажу: если позиция открыта и цена закрытия ниже всех SMA, а объём выше средней величины объёма
    elif global_position == "long":
        if (latest["Close"] < latest["sma_high"] and
            latest["Close"] < latest["sma_medium"] and
            latest["Close"] < latest["sma_low"] and
            latest["Volume"] > latest["vol_ma"]):

            available_asset = get_asset_balance(BASE_ASSET)
            if available_asset < MIN_ASSET_AMOUNT:
                print(f"[{datetime.now()}] Недостаточно актива для продажи: {available_asset} {BASE_ASSET}")
                return

            available_asset = round(available_asset, 6)
            print(f"[{datetime.now()}] Сигнал SELL: Цена {latest['Close']}, Продаём {available_asset} {BASE_ASSET}")

            order = create_order(SYMBOL, "SELL", "MARKET", available_asset)
            print("Ответ ордера на продажу:", order)
            if order and "status" in order and order["status"] == "FILLED":
                global_position = None

# ===============================
# Функции для работы с WebSocket
# ===============================
def process_message(msg: dict):
    """
    Обрабатывает входящие сообщения WebSocket.
    Если получена закрытая свеча (kline), обновляет глобальный DataFrame и проверяет сигналы.
    """
    global global_df

    # Проверяем, что сообщение – это событие kline
    if msg.get("e") != "kline":
        return

    kline = msg["k"]
    is_candle_closed = kline["x"]

    # Извлекаем данные свечи
    open_time = pd.to_datetime(kline["t"], unit="ms")
    open_price = float(kline["o"])
    high_price = float(kline["h"])
    low_price = float(kline["l"])
    close_price = float(kline["c"])
    volume = float(kline["v"])
    close_time = pd.to_datetime(kline["T"], unit="ms")

    if is_candle_closed:
        new_row = {
            "Open Time": open_time,
            "Open": open_price,
            "High": high_price,
            "Low": low_price,
            "Close": close_price,
            "Volume": volume,
            "Close Time": close_time
        }
        global_df = global_df.append(new_row, ignore_index=True)
        # Ограничим размер DataFrame (например, последние 100 свечей)
        if len(global_df) > 100:
            global_df = global_df.iloc[-100:]
        print(f"[{datetime.now()}] Новая закрытая свеча: {open_time} - Цена закрытия: {close_price}")
        update_signals_and_trade()

def on_message(ws, message):
    try:
        msg = json.loads(message)
        process_message(msg)
    except Exception as e:
        print("Ошибка обработки сообщения:", e)

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket закрыт")

def on_open(ws):
    print("WebSocket соединение установлено")

# ===============================
# Основная функция
# ===============================
def main():
    ws_endpoint = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@kline_1m"
    ws = websocket.WebSocketApp(ws_endpoint,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    print("Запуск потоковой торговли...")
    ws.run_forever()

if __name__ == "__main__":
    main()
