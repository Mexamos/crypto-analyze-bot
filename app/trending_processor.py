import logging
import json
import pickle
from asyncio import sleep as async_sleep
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd
from pytz import timezone
from requests.exceptions import HTTPError
from telegram.constants import ParseMode
from telegram.ext import ContextTypes
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from prophet import Prophet
from sentry_sdk import push_scope
import numpy as np

from app.cache.client import CacheClient
from app.database.client import DatabaseClient
from app.crypto.coindesk_client import CoindeskClient
from app.crypto.coingecko_client import CoingeckoClient
from app.crypto.cryptopanic_client import CryptopanicClient
from app.crypto.binance_client import BinanceClient
from app.crypto.newsapi_client import NewsapiClient
from app.crypto.santimentapi_client import SantimentApiClient
from app.crypto.santimentapi_model import ModelTrainingFacade, ModelPredictionFacade
from app.config import Config
from app.monitoring.sentry import SentryClient

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled=True


class TrendingCurrencyProcessor:

    def __init__(
        self,
        coindesk_client: CoindeskClient,
        coingecko_client: CoingeckoClient,
        cryptopanic_client: CryptopanicClient,
        newsapi_client: NewsapiClient,
        santiment_api_client: SantimentApiClient,
        binance_client: BinanceClient,
        model_training_facade: ModelTrainingFacade,
        model_prediction_facade: ModelPredictionFacade,
        cache_client: CacheClient,
        sentry_client: SentryClient,
        config: Config,
        chat_ids: list,
        symbol_to_coingecko_id: dict,
    ):
        self.coindesk_client = coindesk_client
        self.coingecko_client = coingecko_client
        self.cryptopanic_client = cryptopanic_client
        self.newsapi_client = newsapi_client
        self.santiment_api_client = santiment_api_client
        self.binance_client = binance_client
        self.model_training_facade = model_training_facade
        self.model_prediction_facade = model_prediction_facade

        self.cache_client = cache_client
        self.sentry_client = sentry_client
        self.config = config

        self.chat_ids = chat_ids
        self.symbol_to_coingecko_id = symbol_to_coingecko_id
        self.logger = logging.getLogger(__name__)
        self.sentiment_intensity_analyzer = SentimentIntensityAnalyzer()
        self.stopped = False

    async def process(self, context: ContextTypes.DEFAULT_TYPE):
        try:
            trending_coins = self._get_trending_coins()
            coindesk_articles = self._get_coindesk_articles(limit=10)
            cryptopanic_posts = self._get_cryptopanic_posts(limit=50)

            coin_info = await self._forming_currencies_list(
                trending_coins, coindesk_articles, cryptopanic_posts
            )
            new_coin_info = await self._filter_new_records(coin_info)
            if not new_coin_info:
                return

            rows = await self._get_metrics_per_currency(new_coin_info, cryptopanic_posts)

            df = await self._calculate_signals(rows)

            df = df.sort_values("composite", ascending=False)[["symbol", "name", "composite", "signal"]]

            message = await self._generate_message(df.to_dict(), signal_type="BUY")
            if message:
                for chat_id in self.chat_ids:
                    await context.bot.send_message(chat_id, message, parse_mode=ParseMode.MARKDOWN_V2)

        except HTTPError as ex:
            exception_body = ex.response.json()
            with push_scope() as scope:
                scope.set_context("exception_body", exception_body)
                self.sentry_client.capture_exception(ex)
        except Exception as ex:
            self.sentry_client.capture_exception(ex)

    def _get_trending_coins(self):
        response = self.coingecko_client._get_trending_coins()
        return [entry['item'] for entry in response.get('coins', [])]

    def _get_coindesk_articles(self, lang: str = "EN", limit: int = 10) -> list:
        response = self.coindesk_client._get_news_article_list(lang=lang, limit=limit)
        return response.get("Data", [])

    def _get_cryptopanic_posts(self, limit: int = 50) -> list:
        try:
            response = self.cryptopanic_client._get_free_posts(limit)
            return response.get("results", [])
        except HTTPError as exc:
            exception_body = exc.response.json()
            if 'API monthly quota exceeded' in exception_body.get('info', ''):
                return []
            raise exc

    def _get_news_sentiment(self, query: str) -> float:
        try:
            response = self.newsapi_client._get_everything(query)
            articles = response.get("articles", [])
        except HTTPError as exc:
            exception_body = exc.response.json()
            if exception_body.get('code') != 'rateLimited':
                self.sentry_client.capture_exception(
                    Exception(f'NewsAPI failed for {query}: {exception_body}')
                )

            return 0.5

        scores = [
            self.sentiment_intensity_analyzer.polarity_scores(
                article.get("title","") + " " + (article.get("description") or "")
            )["compound"]
            for article in articles
        ]
        return (sum(scores) / len(scores)) if scores else 0.5

    async def _filter_currencies_by_binance(self, currencies: List[dict]) -> List[dict]:
        available_currencies = []
        for currency in currencies:
            try:
                exchange_info = self.binance_client.exchange_info(currency['exchange_symbol'])
                if exchange_info:
                    available_currencies.append(currency)
            except HTTPError as exc:
                try:
                    exception_body = exc.response.json()
                except ValueError:
                    raw = exc.response.text
                    self.sentry_client.capture_exception(
                        Exception(f'Binance error for {currency["exchange_symbol"]}: non-JSON response: {raw!r}')
                    )
                    continue

                if not 'Invalid symbol' in exception_body.get('msg'):
                    self.sentry_client.capture_exception(
                        Exception(f'Get Binance exchange_info failed for {currency["exchange_symbol"]}: {exception_body}')
                    )
                continue

        return available_currencies

    async def _filter_conversion_currency(self, currencies: List[dict]) -> List[dict]:
        available_currencies = []
        for currency in currencies.values():
            if currency['symbol'] != self.config.currency_conversion:
                available_currencies.append(currency)

        return available_currencies

    async def _forming_currencies_list(
        self, trending_coins, coindesk_articles, cryptopanic_posts
    ) -> Dict[str, str]:
        coin_info: dict[str, dict[str, str]] = {}
        seen = set()
        for coin in trending_coins:
            cid = coin['id']
            symbol = coin['symbol'].upper()
            if symbol in seen:
                continue

            coin_info[cid] = {
                'symbol': symbol,
                'exchange_symbol': symbol + self.config.currency_conversion,
                'name': coin['name']
            }
            seen.add(symbol)

        for article in coindesk_articles:
            for asset in article.get('assets', []):
                slug = asset.get('slug', '').lower()
                symbol = slug.upper()
                if not slug or slug in coin_info or symbol in seen:
                    continue

                coin_info[slug] = {
                    'symbol': symbol,
                    'exchange_symbol': symbol + self.config.currency_conversion,
                    'name': asset.get('name', slug)
                }
                seen.add(symbol)

        for post in cryptopanic_posts:
            for cur in post.get('currencies', []):
                slug = cur.get('code', '').lower()
                symbol = slug.upper()
                if not slug or slug in coin_info or symbol in seen:
                    continue

                coin_info[slug] = {
                    'symbol': symbol,
                    'exchange_symbol': symbol + self.config.currency_conversion,
                    'name': symbol
                }
                seen.add(symbol)

        coin_info = await self._filter_conversion_currency(coin_info)
        return await self._filter_currencies_by_binance(coin_info)

    async def _filter_new_records(self, records: dict[str, dict[str, str]]) -> List[dict[str, str]]:
        records = [f"{record['symbol']}:{record['exchange_symbol']}:{record['name']}" for record in records]
        cached_records = self.cache_client.get('last_trending_currencies')
        new_records = []
        if cached_records:
            last_records = json.loads(cached_records)
            for record in records:
                if record not in last_records:
                    record = record.split(':')
                    new_records.append({
                        'symbol': record[0],
                        'exchange_symbol': record[1],
                        'name': record[2]
                    })

        self.cache_client.set('last_trending_currencies', json.dumps(records))

        return new_records

    async def _get_historical_klines(self, symbol: str, interval: str = '1h', limit: int = 1000) -> pd.DataFrame:
        """Fetch historical klines (OHLCV) data from Binance"""
        try:
            klines = self.binance_client.klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )

            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close Time', 'Quote Asset Volume', 'Number of Trades',
                'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
            ])

            # Convert types
            df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
            df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

        except Exception as e:
            self.sentry_client.capture_exception(e)
            return pd.DataFrame()

    async def _get_metrics_per_currency(self, coin_info, cryptopanic_posts):
        rows: List[dict[str, float | str]] = []
        for info in coin_info:
            if self.stopped:
                break

            symbol = info['symbol']
            name = info['name']
            exchange_symbol = info['exchange_symbol']
            coingecko_id = self.symbol_to_coingecko_id.get(symbol)

            try:
                # Get historical OHLCV data
                historical_data = await self._get_historical_klines(exchange_symbol, limit=200)

                comm_score = await self._get_coin_sentiment(coingecko_id)
            except Exception as ex:
                with push_scope() as scope:
                    scope.set_context("symbol", symbol)
                    scope.set_context("exchange_symbol", exchange_symbol)
                    scope.set_context("coingecko_id", coingecko_id)
                    self.sentry_client.capture_exception(ex)

                raise ex

            news_score = self._get_news_sentiment(name)

            cp_posts = [
                p for p in cryptopanic_posts
                if any(c.get('code', '').lower() == symbol.lower() for c in p.get('currencies', []))
            ]
            cp_score = sum(
                p.get('votes', {}).get('positive', 0) - p.get('votes', {}).get('negative', 0)
                for p in cp_posts
            )

            row_data = {
                "symbol": symbol,
                "name": name,
                "comm_score": comm_score,
                "news_score": news_score,
                "cryptopanic_score": cp_score,
            }

            # Add OHLCV data if available
            if not historical_data.empty:
                row_data.update({
                    "Open Time": historical_data['Open Time'].iloc[-1],
                    "Open": historical_data['Open'].iloc[-1],
                    "High": historical_data['High'].iloc[-1],
                    "Low": historical_data['Low'].iloc[-1],
                    "Close": historical_data['Close'].iloc[-1],
                    "Volume": historical_data['Volume'].iloc[-1],
                    "historical_data": historical_data
                })

            rows.append(row_data)

        return rows

    async def _get_coin_sentiment(
        self, coin_id: str, max_retries: int = 3, backoff_factor: float = 1.0
    ) -> float:
        for attempt in range(1, max_retries + 1):
            if self.stopped:
                break

            try:
                response = self.coingecko_client._get_coin_by_id(coin_id)
                up_pct = response.get('sentiment_votes_up_percentage', 0) or 0
                down_pct = response.get('sentiment_votes_down_percentage', 0) or 0

                total = up_pct + down_pct
                return (up_pct / total) if total else 0.5
            except HTTPError as exc:
                status = getattr(exc.response, 'status_code', None)
                if status == 429:
                    retry_after = exc.response.headers.get('Retry-After')
                    wait_time = (
                        int(retry_after) if retry_after and retry_after.isdigit() else 60
                    ) + backoff_factor * (attempt - 1)
                    await async_sleep(wait_time)
                    continue

                self.sentry_client.capture_exception(Exception(f'[{coin_id}] HTTP {status}: {exc}, returning 0.5'))
                return 0.5

        self.sentry_client.capture_exception(Exception(f'[{coin_id}] Failed {max_retries} attempts, returning 0.5'))
        return 0.5

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for trading decisions"""
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']

        # Bollinger Bands
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['BB_Upper'] = df['SMA_20'] + (df['Close'].rolling(window=20).std() * 2)
        df['BB_Lower'] = df['SMA_20'] - (df['Close'].rolling(window=20).std() * 2)

        # Volume Analysis
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']

        return df

    def _generate_technical_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on technical indicators"""
        signals = pd.DataFrame(index=df.index)
        
        # RSI signals
        signals['RSI_Signal'] = 0
        signals.loc[df['RSI'] < 30, 'RSI_Signal'] = 1  # Oversold - Buy
        signals.loc[df['RSI'] > 70, 'RSI_Signal'] = -1  # Overbought - Sell

        # MACD signals
        signals['MACD_Signal'] = 0
        signals.loc[df['MACD'] > df['Signal_Line'], 'MACD_Signal'] = 1
        signals.loc[df['MACD'] < df['Signal_Line'], 'MACD_Signal'] = -1

        # Bollinger Bands signals
        signals['BB_Signal'] = 0
        signals.loc[df['Close'] < df['BB_Lower'], 'BB_Signal'] = 1  # Price below lower band - Buy
        signals.loc[df['Close'] > df['BB_Upper'], 'BB_Signal'] = -1  # Price above upper band - Sell

        # Volume signals
        signals['Volume_Signal'] = 0
        signals.loc[df['Volume_Ratio'] > 1.5, 'Volume_Signal'] = 1  # High volume - potential trend
        signals.loc[df['Volume_Ratio'] < 0.5, 'Volume_Signal'] = -1  # Low volume - potential reversal

        # Combine signals
        signals['Technical_Score'] = (
            signals['RSI_Signal'] * 0.3 +
            signals['MACD_Signal'] * 0.3 +
            signals['BB_Signal'] * 0.2 +
            signals['Volume_Signal'] * 0.2
        )

        return signals

    def _prepare_prophet_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Prophet model"""
        prophet_df = df[['Open Time', 'Close']].copy()
        prophet_df.columns = ['ds', 'y']
        return prophet_df

    def _train_prophet_model(self, df: pd.DataFrame, symbol: str) -> Prophet:
        cache_key = f"prophet_model_{symbol}"
        cached_model = self.cache_client.get(cache_key)
        if cached_model:
            try:
                model = pickle.loads(cached_model)
                return model
            except Exception as e:
                self.sentry_client.capture_exception(e)
        
        prophet_df = self._prepare_prophet_data(df)
        
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            growth='linear',
            n_changepoints=5,  # Allow some changepoints for better trend detection
            changepoint_range=0.8,  # Use 80% of the data for changepoints
            mcmc_samples=0,
            stan_backend='CMDSTANPY',
            interval_width=0.95,  # 95% prediction intervals
            seasonality_mode='multiplicative'  # Better for financial data
        )
        # Add custom seasonality if needed
        model.add_seasonality(
            name='hourly',
            period=24,
            fourier_order=5
        )
        model.fit(prophet_df)

        try:
            serialized_model = pickle.dumps(model)
            self.cache_client.set(cache_key, serialized_model, ex=3600)  # 3600 seconds = 1 hour
        except Exception as e:
            self.sentry_client.capture_exception(e)

        return model

    def _get_price_prediction(self, model: Prophet, periods: int = 24) -> pd.DataFrame:
        """Get price predictions from Prophet model"""
        future = model.make_future_dataframe(periods=periods, freq='h')  # Changed from 'H' to 'h'
        forecast = model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def _calculate_position_size(self, df: pd.DataFrame, available_balance: float) -> float:
        """Calculate position size based on volatility and risk parameters"""
        # Calculate ATR for volatility
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]
        
        # Calculate volatility-based position size
        risk_per_trade = 0.02  # 2% risk per trade
        stop_loss_atr_multiplier = 2  # Stop loss at 2 ATR
        
        # Calculate position size
        risk_amount = available_balance * risk_per_trade
        stop_loss_distance = atr * stop_loss_atr_multiplier
        position_size = risk_amount / stop_loss_distance
        
        # Limit position size to 20% of available balance
        max_position = available_balance * 0.2
        position_size = min(position_size, max_position)
        
        return position_size

    def _calculate_trailing_stop(self, df: pd.DataFrame, entry_price: float) -> float:
        """Calculate trailing stop loss based on ATR and entry price"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]
        
        # Calculate highest price since entry
        highest_price = max(entry_price, df['High'].iloc[-1])
        
        # Set trailing stop at 2 ATR below the highest price since entry
        # This ensures we don't set the stop loss below our entry price
        trailing_stop = max(
            entry_price,  # Don't go below entry price
            highest_price - (2 * atr)  # Standard trailing stop calculation
        )
        
        return trailing_stop

    async def _calculate_signals(self, rows) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        
        # Calculate technical indicators if we have historical data
        if 'historical_data' in df.columns:
            # Process each row's historical data
            for idx, row in df.iterrows():
                if isinstance(row['historical_data'], pd.DataFrame):
                    # Calculate technical indicators for this symbol's historical data
                    symbol_data = self._calculate_technical_indicators(row['historical_data'])
                    technical_signals = self._generate_technical_signals(symbol_data)

                    # Get the last values
                    df.at[idx, 'technical_score'] = technical_signals['Technical_Score'].iloc[-1]
                    print(f"Technical score for {row['symbol']}: {df.at[idx, 'technical_score']}")

                    # Add Prophet predictions
                    try:
                        prophet_model = self._train_prophet_model(symbol_data, row['symbol'])
                        forecast = self._get_price_prediction(prophet_model, periods=1)
                        
                        # Calculate prediction confidence
                        last_price = symbol_data['Close'].iloc[-1]
                        next_prediction = forecast['yhat'].iloc[-1]
                        prediction_range = forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1]
                        
                        # Normalize prediction confidence
                        df.at[idx, 'prediction_score'] = 1 - (prediction_range / last_price)
                        print(f"Prediction score for {row['symbol']}: {df.at[idx, 'prediction_score']}")
                        
                        # Add trend prediction
                        price_change = (next_prediction - last_price) / last_price
                        df.at[idx, 'trend_prediction'] = np.clip(price_change * 5, -1, 1)
                        print(f"Trend prediction for {row['symbol']}: {df.at[idx, 'trend_prediction']}")
                        
                        # Calculate position size and risk metrics
                        available_balance = float(self.config.total_available_amount)
                        df.at[idx, 'position_size'] = self._calculate_position_size(symbol_data, available_balance)
                        df.at[idx, 'trailing_stop'] = self._calculate_trailing_stop(symbol_data, last_price)
                        
                    except Exception as e:
                        self.sentry_client.capture_exception(e)
                        df.at[idx, 'prediction_score'] = 0
                        df.at[idx, 'trend_prediction'] = 0
                        df.at[idx, 'position_size'] = 0
                        df.at[idx, 'trailing_stop'] = 0

                print(' ')
        else:
            df['technical_score'] = 0
            df['prediction_score'] = 0
            df['trend_prediction'] = 0
            df['position_size'] = 0
            df['trailing_stop'] = 0

        # Normalize sentiment scores
        for col in ["comm_score", "news_score", "cryptopanic_score"]:
            mn, mx = df[col].min(), df[col].max()
            df[f"{col}_n"] = (df[col] - mn) / (mx - mn) if mx > mn else 0.5

        # Compute composite score with all factors
        df["composite"] = (
            0.1 * df["comm_score_n"] +
            0.1 * df["news_score_n"] +
            0.1 * df["cryptopanic_score_n"] +
            0.25 * df["technical_score"] +
            0.25 * df["prediction_score"] +
            0.2 * df["trend_prediction"]
        )

        # Generate signals with dynamic thresholds and risk management
        def generate_signal(row):
            if row['composite'] > 0.6 and row['position_size'] > 0:
                return "BUY"
            elif row['composite'] < 0.3 or (row['Close'] < row['trailing_stop'] if 'Close' in row else False):
                return "SELL"
            return "HOLD"
            
        df["signal"] = df.apply(generate_signal, axis=1)
        
        # Clean up the DataFrame before returning
        df = df.drop('historical_data', axis=1, errors='ignore')
        
        return df

    async def _generate_message(self, records, signal_type: Optional[str] = None):
        signals = [
            (symbol, name, composite, signal)
            for symbol, name, composite, signal in zip(
                records['symbol'].values(),
                records['name'].values(),
                records['composite'].values(),
                records['signal'].values()
            )
            if signal_type and signal == signal_type or not signal_type
        ]
        
        if not signals:
            return None

        message_lines = [
            f"{symbol} ({name})" + ((30 - len(f"{symbol} ({name})")) * ' ') + f"{composite:.3f} {signal}"
            for symbol, name, composite, signal in signals
        ]

        return '```\n' + "\n".join(message_lines) + '```'

    def _get_santiment_api_trend_words(self):
        to_date = datetime.now()
        from_date = to_date - timedelta(days=1)

        trend_words = self.santiment_api_client.emerging_trends(
            from_date=from_date.isoformat(),
            to_date=to_date.isoformat(),
        )
        valid_trend_words = []
        for trend_word in trend_words:
            if trend_word.upper() in self.symbol_to_coingecko_id.keys():
                valid_trend_words.append(trend_word.upper())

        return valid_trend_words
