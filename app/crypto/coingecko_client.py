from datetime import datetime
from typing import Optional

from pytz import timezone
from requests import Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects

from app.config import Config


class CoingeckoException(Exception):
    pass


class CoingeckoClient:

    def __init__(self, host: str, api_key: str, config: Config) -> None:
        self.host = host
        self.api_key = api_key

        self.config = config
        self.timezone = timezone(self.config.timezone_name)
        self.latest_request_datetime = None

    def _call(self, url: str, params: Optional[dict] = None):
        url = self.host + url
        headers = {
            'accept': 'application/json',
            'x-cg-api-key': self.api_key
        }
        session = Session()
        session.headers.update(headers)

        try:
            response = session.get(url, params=params)
            response.raise_for_status()
            json_result = response.json()

            self.latest_request_datetime = datetime.now(self.timezone)
            return json_result
        except (ConnectionError, Timeout, TooManyRedirects) as ex:
            raise CoingeckoException() from ex

    def _get_coins_list(self):
        url = '/api/v3/coins/list'
        return self._call(url)

    def _get_trending_coins(self):
        url = '/api/v3/search/trending'
        return self._call(url)

    def _get_coin_by_id(self, coin_id: str):
        url = f'/api/v3/coins/{coin_id}'
        params = {
            "localization":   "false",
            "tickers":        "false",
            "market_data":    "false",
            "community_data": "false",
            "developer_data": "false",
            "sparkline":      "false",
        }
        return self._call(url, params)
    
    def _get_current_price(self, coin_id, vs_currency: str):
        url = f"/api/v3/simple/price"
        params = {
            "ids": coin_id,
            "vs_currencies": vs_currency
        }
        return self._call(url, params)
