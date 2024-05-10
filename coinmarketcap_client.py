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
# текущий date_time


# Trending Gainers & Losers
# https://coinmarketcap.com/api/documentation/v1/#operation/getV1CryptocurrencyTrendingGainerslosers

# Trending Latest
# https://coinmarketcap.com/api/documentation/v1/#operation/getV1CryptocurrencyTrendingLatest

from typing import List

from requests import Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects


class CmcException(Exception):
    pass


class CoinmarketcapClient:

    def __init__(self, api_key: str) -> None:
        self.host = 'https://pro-api.coinmarketcap.com'
        self.api_key = api_key

    def _trending_latest(self):
        url = self.host + '/v1/cryptocurrency/trending/latest'
        parameters = {
            'start':'1',
            'limit':'100',
            'time_period': '24h',
        }
        headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': self.api_key,
        }

        session = Session()
        session.headers.update(headers)

        try:
            response = session.get(url, params=parameters)
            return response.json()
        except (ConnectionError, Timeout, TooManyRedirects) as ex:
            raise CmcException(str(ex))

    def _filter_data(self, currency):
            if currency['quote']['USD']['percent_change_24h'] > 0:
                return True 
            else:
                return False

    def _sort_data(self, currency):
            return currency['quote']['USD']['percent_change_24h']

    def actual_trending_latest_currencies(self) -> List[dict]:
        data = self._trending_latest()

        filtered_data = filter(self._filter_data, data['data'])
        sorted_data = sorted(filtered_data, key=self._sort_data, reverse=True)

        return sorted_data
