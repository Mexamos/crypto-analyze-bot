from unittest.mock import MagicMock

import pytest

from app.bot_controller import BotController
from app.database.client import DatabaseClient
from app.crypto.coinmarketcap_client import CoinmarketcapClient
from app.crypto.binance_client import BinanceClient
from app.analytics.chart import ChartController
from app.config import Config
from app.analytics.google_sheets_client import GoogleSheetsClient
from app.monitoring.sentry import SentryClient


@pytest.fixture
def fake_db_client():
    return DatabaseClient()


@pytest.fixture
def fake_config():
    return Config()


@pytest.fixture
def fake_cmc_client(fake_config):
    return CoinmarketcapClient('api_key', fake_config)


@pytest.fixture
def fake_binance_cleint():
    return BinanceClient('api_key', 'secret_key')


@pytest.fixture
def fake_chart_controller(fake_db_client, fake_config):
    return ChartController(fake_db_client, fake_config)


@pytest.fixture
def fake_google_sheets_client():
    return MagicMock()


@pytest.fixture
def fake_sentry_client():
    return MagicMock()


@pytest.fixture
def fake_bot_controller(
    fake_db_client: DatabaseClient, fake_cmc_client: CoinmarketcapClient,
    fake_binance_cleint: BinanceClient, fake_chart_controller: ChartController,
    fake_google_sheets_client, fake_sentry_client, fake_config: Config
):
    return BotController(
        fake_db_client, fake_cmc_client, fake_binance_cleint,
        fake_chart_controller, fake_google_sheets_client,
        fake_sentry_client, fake_config, 'bot_token', chat_id='100'
    )


@pytest.fixture
def fake_context():
    return MagicMock()


@pytest.fixture
def mock_trending_gainers_losers(mocker):

    def inner(return_value=None):
        mocker.patch(
            'app.crypto.coinmarketcap_client.CoinmarketcapClient._trending_gainers_losers',
            return_value=return_value
        )

    return inner


@pytest.fixture
def mock_filter_currencies_by_binance(mocker):

    def inner(return_value=None):
        mocker.patch(
            'app.bot_controller.BotController._filter_currencies_by_binance',
            return_value=return_value
        )

    return inner


@pytest.fixture
def mock_get_quotes_historical(mocker):

    def inner(return_value=None):
        mocker.patch(
            'app.crypto.coinmarketcap_client.CoinmarketcapClient.get_quotes_historical',
            return_value=return_value
        )

    return inner


@pytest.fixture
def mock_get_quotes_latest(mocker):

    def inner(return_value=None):
        mocker.patch(
            'app.crypto.coinmarketcap_client.CoinmarketcapClient.get_quotes_latest',
            return_value=return_value
        )

    return inner
