from datetime import datetime
from typing import List, Optional

from pytz import timezone
from requests import Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects

from app.config import Config


class CmcException(Exception):
    pass


class CoinmarketcapClient:

    def __init__(self, api_key: str, config: Config) -> None:
        self.host = 'https://pro-api.coinmarketcap.com'
        self.api_key = api_key

        self.config = config
        self.timezone = timezone(self.config.timezone_name)
        self.latest_request_datetime = None

    def _call(self, url: str, params: dict):
        url = self.host + url
        headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': self.api_key,
        }

        session = Session()
        session.headers.update(headers)

        try:
            response = session.get(url, params=params)
            json_result = response.json()

            self.latest_request_datetime = datetime.now(self.timezone)
            return json_result
        except (ConnectionError, Timeout, TooManyRedirects) as ex:
            raise CmcException() from ex
        
    def get_id_map(self, symbol: Optional[str] = None) -> List[dict]:
        url = '/v1/cryptocurrency/map'
        parameters = {
            'sort': 'cmc_rank',
        }
        if symbol:
            parameters['symbol'] = symbol

        return self._call(url, parameters)

    def _trending_latest(self):
        url = '/v1/cryptocurrency/trending/latest'
        parameters = {
            'start':'1',
            'limit':'100',
            'time_period': '24h',
        }
        return self._call(url, parameters)

    def _trending_gainers_losers(self):
        url = '/v1/cryptocurrency/trending/gainers-losers'
        parameters = {
            'start':'1',
            'limit':'100',
            'time_period': '24h',
        }
        return self._call(url, parameters)

    def _filter_data(self, currency):
        if currency['quote']['USD']['percent_change_24h'] > 0:
            return True 
        else:
            return False

    def _sort_data(self, currency):
        return currency['quote']['USD']['percent_change_24h']

    def get_trending_currencies(self) -> List[dict]:
        data = self._trending_gainers_losers()

        filtered_data = filter(self._filter_data, data['data'])
        sorted_data = sorted(filtered_data, key=self._sort_data, reverse=True)

        return sorted_data

    def get_quotes_latest(self, cmc_id: int, convert: Optional[str] = None):
        url = '/v2/cryptocurrency/quotes/latest'
        parameters = {
            'id': cmc_id,
        }
        if convert:
            parameters['convert'] = convert

        return self._call(url, parameters)

    def get_quotes_historical(self, cmc_id: int):
        url = '/v2/cryptocurrency/quotes/historical'
        parameters = {
            'id': cmc_id,
            'count': self.config.quotes_historical_count,
            'interval': self.config.quotes_historical_interval,
        }
        return self._call(url, parameters)
