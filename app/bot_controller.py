import json
import math
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
from logging.handlers import RotatingFileHandler
from time import sleep

import pandas as pd
import numpy as np
from pandas import DataFrame
from redis import Redis
from websocket import WebSocketApp
from websocket._exceptions import WebSocketConnectionClosedException

from app.crypto.binance_client import BinanceClient
from app.config import Config
from app.monitoring.sentry import SentryClient


class BotController:

    def __init__(
        self, config: Config, binance_cleint: BinanceClient, redis_client: Redis
    ) -> None:
        self.bot_is_running = True
        self.restart_time = 30

        self.init_logs(config.logs_file_path)

        self.base_asset = config.base_asset
        self.trade_asset = config.trade_asset
        self.symbol = config.symbol
        self.minimum_trade_amount = config.minimum_trade_amount
        self.ws_endpoint = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@kline_1s"

        self.binance_cleint = binance_cleint
        self.redis_client = redis_client

        self.position = config.position
        self.window_high = config.window_high
        self.window_medium = config.window_medium
        self.window_low = config.window_low
        self.vol_window = config.vol_window
        self.adx_threshold = config.adx_threshold
        self.atr_period = config.atr_period
        self.atr_stop_multiplier = config.atr_stop_multiplier
        self.stop_loss = None
        self.quantity_precession = 0

    def init_logs(self, logs_file_path: str):
        self.logger = getLogger('bot')
        self.logger.setLevel(INFO)
        formatter = Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%d.%m.%Y %H:%M:%S'
        )

        stream_handler = StreamHandler()
        stream_handler.setLevel(INFO)
        stream_handler.setFormatter(formatter)

        file_handler = FileHandler(logs_file_path)
        file_handler.setLevel(INFO)
        file_handler.setFormatter(formatter)

        rotating_file_handler = RotatingFileHandler(logs_file_path, maxBytes=1000000, backupCount=5)  # 1 MB

        self.logger.addHandler(rotating_file_handler)
        self.logger.addHandler(stream_handler)
        self.logger.addHandler(file_handler)

    def request_static_data(self):
        response = self.binance_cleint.exchange_info(self.symbol)
        symbol_data = response['symbols'][0]
        for filter in symbol_data['filters']:
            if filter['filterType'] == 'LOT_SIZE':
                self.step_size = float(filter['stepSize'])
                self.quantity_precession = len(str(filter['stepSize']).split('.')[1])

    def run_bot(self):
        # Run the bot until the user presses Ctrl-C
        while self.bot_is_running:
            ws = WebSocketApp(
                self.ws_endpoint,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
            )
            self.logger.info("Запуск потоковой торговли...")
            ws.run_forever(ping_interval=20, ping_timeout=10)

    def on_message(self, ws, message):
        try:
            msg = json.loads(message)
            self.process_message(msg)
        except WebSocketConnectionClosedException as exc:
            ws.close_exc = exc
            ws.close()
        except BaseException as exc:
            exc_data = exc
            if hasattr(exc, 'response') and hasattr(exc.response, 'json'):
                exc_data = exc.response.json()

            self.logger.exception(f'Ошибка обработки сообщения: {exc_data}')
            ws.close()

    def on_error(self, ws, error):
        if type(error) not in (KeyboardInterrupt, WebSocketConnectionClosedException):
            self.logger.exception(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        self.logger.info("WebSocket закрыт")

        close_exc = getattr(ws, 'close_exc', None)
        if not isinstance(close_exc, WebSocketConnectionClosedException):
            self.bot_is_running = False
        else:
            self.logger.error(f"Ошибка WebSocket: {close_exc}. Попытка переподключения через {self.restart_time} секунд...")
            sleep(self.restart_time)

    def on_open(self, ws):
        self.logger.info("WebSocket соединение установлено")

    def process_message(self, msg: dict):
        """
        Обрабатывает входящие сообщения WebSocket.
        Если получена закрытая свеча (kline), обновляет глобальный DataFrame и проверяет сигналы.
        """
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
                "Open Time": open_time.isoformat(),
                "Open": open_price,
                "High": high_price,
                "Low": low_price,
                "Close": close_price,
                "Volume": volume,
                "Close Time": close_time.isoformat()
            }

            self.redis_client.rpush("candles", json.dumps(new_row))
            # Ограничиваем список последних 100 свечей
            self.redis_client.ltrim("candles", -100, -1)

            self.update_signals_and_trade()

    def parse_datetime(self, data):
        if '.' in data:
            return pd.to_datetime(data, format="%Y-%m-%dT%H:%M:%S.%f")
        else:
            return pd.to_datetime(data, format="%Y-%m-%dT%H:%M:%S")

    def calculate_indicators(self, df: DataFrame) -> DataFrame:
        # Преобразуем поля времени в datetime
        df["Open Time"] = df["Open Time"].apply(self.parse_datetime)
        df["Close Time"] = df["Close Time"].apply(self.parse_datetime)

        # Вычисление скользящих средних для цены закрытия
        df["sma_high"] = df["Close"].rolling(window=self.window_high).mean()
        df["sma_medium"] = df["Close"].rolling(window=self.window_medium).mean()
        df["sma_low"] = df["Close"].rolling(window=self.window_low).mean()
        # Скользящая средняя для объёма
        df["vol_ma"] = df["Volume"].rolling(window=self.vol_window).mean()

        # --- Вычисляем ATR ---
        df["H-L"] = df["High"] - df["Low"]
        df["H-PC"] = (df["High"] - df["Close"].shift(1)).abs()
        df["L-PC"] = (df["Low"] - df["Close"].shift(1)).abs()
        df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
        df["atr"] = df["TR"].rolling(window=self.atr_period).mean()

        # --- Вычисляем ADX ---
        df["up_move"] = df["High"] - df["High"].shift(1)
        df["down_move"] = df["Low"].shift(1) - df["Low"]
        df["+DM"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0)
        df["-DM"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0)
        df["TR_sum"] = df["TR"].rolling(window=self.atr_period).sum()
        df["+DM_sum"] = df["+DM"].rolling(window=self.atr_period).sum()
        df["-DM_sum"] = df["-DM"].rolling(window=self.atr_period).sum()
        df["+DI"] = 100 * (df["+DM_sum"] / df["TR_sum"])
        df["-DI"] = 100 * (df["-DM_sum"] / df["TR_sum"])
        df["DX"] = 100 * (abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"]))
        df["adx"] = df["DX"].rolling(window=self.atr_period).mean()

        return df

    def round_step_size_with_precision(self, quantity, decimals):
        # First, round down to the nearest valid multiple of step_size
        valid_quantity = math.floor(quantity / self.step_size) * self.step_size
        # Then, round down the result to the desired precision
        factor = 10 ** decimals
        return math.floor(valid_quantity * factor) / factor

    def update_signals_and_trade(self):
        """
        Рассчитывает скользящие средние и, если условия выполнены, отправляет ордера.
        """
        # Для расчёта индикаторов нужно минимум WINDOW_HIGH свечей
        candles_data = self.redis_client.lrange("candles", 0, -1)
        if len(candles_data) < self.window_high:
            self.logger.info(f'Not enough data. Current: {len(candles_data)}, requires: {self.window_high}')
            return

        data_list = [json.loads(item.decode('utf-8')) for item in candles_data]
        df = DataFrame(data_list)

        df = self.calculate_indicators(df)

        # Берём самую последнюю свечу
        latest = df.iloc[-1]

        # --- Логика сигналов ---
        # Сигнал на покупку: если позиции нет и цена закрытия выше всех SMA, а объём выше средней величины объёма
        if (
            not self.position and
            latest["adx"] >= self.adx_threshold and
            latest["Close"] > latest["sma_high"] and
            latest["Close"] > latest["sma_medium"] and
            latest["Close"] > latest["sma_low"] and
            latest["Volume"] > latest["vol_ma"]
        ):
            # Получаем баланс USDT
            available_usdt = self.get_asset_balance(self.base_asset)
            self.logger.info(f'Доступное количество USDT: {available_usdt}')
            if available_usdt < self.minimum_trade_amount:
                self.logger.info(f'Недостаточно средств для покупки: {available_usdt} USDT. Минимум: {self.minimum_trade_amount}')
                return

            # Используем доступные USDT как сумму для покупки (quoteOrderQty)
            self.logger.info(f"Сигнал BUY: Цена {latest['Close']}, Покупаем на сумму {available_usdt} {self.base_asset}")

            order = self.binance_cleint.make_order(self.symbol, "BUY", "MARKET", quoteOrderQty=available_usdt)
            self.logger.info(f"Ответ ордера на покупку: {order}")
            if order and "status" in order and order["status"] == "FILLED":
                self.position = "long"

                # Сохраняем цену входа и рассчитываем уровень стоп-лосса по ATR
                self.entry_price = latest["Close"]
                if not pd.isna(latest["atr"]):
                    self.stop_loss = self.entry_price - self.atr_stop_multiplier * latest["atr"]
                else:
                    self.stop_loss = None

        # Сигнал на продажу: если позиция открыта и цена закрытия ниже всех SMA, а объём выше средней величины объёма
        elif self.position == "long":
            sell = False
            if self.stop_loss is not None and latest["Low"] < self.stop_loss:
                self.logger.info(f"Stop-loss сработал: Latest low {latest['Low']} < stop_loss {self.stop_loss}")
                sell = True
            elif (
                latest["Close"] < latest["sma_high"] and
                latest["Close"] < latest["sma_medium"] and
                latest["Close"] < latest["sma_low"] and
                latest["Volume"] > latest["vol_ma"]
            ):
                sell = True

            if sell:
                available_asset = self.get_asset_balance(self.trade_asset)
                quantity = self.round_step_size_with_precision(available_asset, self.quantity_precession)
                self.logger.info(f"Сигнал SELL: Цена {latest['Close']}, Продаём {quantity} {self.trade_asset}")

                order = self.binance_cleint.make_order(self.symbol, "SELL", "MARKET", quantity=quantity)
                self.logger.info(f"Ответ ордера на продажу: {order}")
                if order and "status" in order and order["status"] == "FILLED":
                    self.position = None
                    self.entry_price = None
                    self.stop_loss = None

    def get_asset_balance(self, asset: str) -> float:
        return self.binance_cleint.get_account_balance(asset)
