from typing import Optional

import requests
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects


class NewsapiException(Exception):
    pass


class NewsapiClient:

    def __init__(self, host: str, api_key: str) -> None:
        self.host = host
        self.api_key = api_key

    def _call(self, url: str, params: Optional[dict] = None):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except (ConnectionError, Timeout, TooManyRedirects) as ex:
            raise NewsapiException() from ex

    def _get_everything(self, query: str):
        url = self.host + '/v2/everything'
        params = {
            'q': query, 'pageSize': 10, 'apiKey': self.api_key
        }
        return self._call(url, params)
