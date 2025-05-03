from typing import Optional

import requests
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects


class CoindeskException(Exception):
    pass


class CoindeskClient:

    def __init__(self, host: str) -> None:
        self.host = host

    def _call(self, url: str, headers: Optional[dict] = None, params: Optional[dict] = None):
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except (ConnectionError, Timeout, TooManyRedirects) as ex:
            raise CoindeskException() from ex

    def _get_news_article_list(self, lang: str = "EN", limit: int = 10):
        url = self.host + '/news/v1/article/list'
        headers = {
            "Content-type": "application/json; charset=UTF-8"
        }
        params = {
            "lang": lang, "limit": limit
        }

        return self._call(url, headers, params)
