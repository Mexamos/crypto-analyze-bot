from typing import Optional

import requests
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects


class CryptopanicException(Exception):
    pass


class CryptopanicClient:

    def __init__(self, host: str, auth_token: str) -> None:
        self.host = host
        self.auth_token = auth_token

    def _call(self, url: str, params: Optional[dict] = None):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except (ConnectionError, Timeout, TooManyRedirects) as ex:
            raise CryptopanicException() from ex

    def _get_free_posts(self, limit: int = 50):
        url = self.host + '/api/v1/posts/'
        params = {
            'auth_token': self.auth_token,
            'filter': 'hot',
            'public': 'true',
            'limit': limit
        }
        return self._call(url, params)
