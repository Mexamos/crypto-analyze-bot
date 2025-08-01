import hmac
import hashlib
from typing import Optional, List
from time import time
import urllib.parse

import requests


class BinanceClient:

    def __init__(self, api_key: str, secret_key: str) -> None:
        self.host = 'https://api.binance.com'
        self.api_key = api_key
        self.secret_key = secret_key

    def _call(self, url: str, params: dict, method: str = 'GET'):
        headers = {
            'Content-Type': 'application/json',
            'X-MBX-APIKEY': self.api_key
        }
        response = requests.request(method, url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def exchange_info(self, symbol: str):
        url = self.host + '/api/v3/exchangeInfo'
        params = {
            'symbol': symbol
        }
        return self._call(url, params)

    def klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = 1000,
    ):
        """
            # Interval	interval value
            # seconds	1s
            # minutes	1m, 3m, 5m, 15m, 30m
            # hours	1h, 2h, 4h, 6h, 8h, 12h
            # days	1d, 3d
            # weeks	1w
            # months	1M
        """
        url = self.host + '/api/v3/klines'
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit,
        }

        if start_time:
            params['startTime'] = int(start_time.timestamp()) * 1000

        if end_time:
            params['endTime'] = int(end_time.timestamp()) * 1000

        return self._call(url, params)

    def _generate_signature(self, query_string: str):
        hashing_object = hmac.new(
            self.secret_key.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256
        )
        signature = hashing_object.hexdigest()
        return signature

    def get_account_balance(
        self, assets: Optional[List[str]] = None
    ):
        timestamp = int(time() * 1000)
        query_string = f'timestamp={timestamp}'
        signature = self._generate_signature(query_string)

        url = f'{self.host}/api/v3/account?{query_string}&signature={signature}'
        response = self._call(url, {})

        balances = response['balances']
        if assets:
            return {balance['asset']: float(balance['free']) for balance in balances if balance['asset'] in assets}

        return balances

    def get_account_trade_list(self, asset: str):
        timestamp = int(time() * 1000)
        query_string = f'timestamp={timestamp}&symbol={asset}'
        signature = self._generate_signature(query_string)

        url = f'{self.host}/api/v3/myTrades?{query_string}&signature={signature}'
        response = self._call(url, {})
        return response
    
    def get_account_convert_history(
        self, start_time: int, end_time: int
    ):
        timestamp = int(time() * 1000)
        params = {
            'startTime': int(start_time.timestamp()) * 1000,
            'endTime': int(end_time.timestamp()) * 1000,
            'timestamp': timestamp
        }
        query_string = urllib.parse.urlencode(params)
        signature = self._generate_signature(query_string)
        
        url = f'{self.host}/sapi/v1/convert/tradeFlow?{query_string}&signature={signature}'
        response = self._call(url, {})

        return response

    def get_all_orders(self, asset: str):
        timestamp = int(time() * 1000)
        query_string = f'timestamp={timestamp}&symbol={asset}'
        signature = self._generate_signature(query_string)

        url = f'{self.host}/api/v3/allOrders?{query_string}&signature={signature}'
        response = self._call(url, {})
        return response

    def make_order(
        self, symbol: str, side: str, order_type: str,
        quoteOrderQty: Optional[float] = None, quantity: Optional[float] = None
    ):
        timestamp = int(time() * 1000)
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'timestamp': timestamp
        }
        if quoteOrderQty:
            params['quoteOrderQty'] = quoteOrderQty
        else:
            params['quantity'] = quantity

        query_string = urllib.parse.urlencode(params)
        signature = self._generate_signature(query_string)

        url = f'{self.host}/api/v3/order?{query_string}&signature={signature}'

        return self._call(url, {}, method='POST')

    def make_convertation(self, from_asset: str, to_asset: str, from_amount: str):
        timestamp = int(time() * 1000)
        params = {
            'fromAsset': from_asset,
            'toAsset': to_asset,
            'fromAmount': from_amount,
            'timestamp': timestamp
        }
        query_string = urllib.parse.urlencode(params)
        signature = self._generate_signature(query_string)
        url = f'{self.host}/sapi/v1/convert/getQuote?{query_string}&signature={signature}'
        response = self._call(url, {}, method='POST')
        print('response 1', response)
        quote_id = response['quoteId']

        
        timestamp = int(time() * 1000)
        params = {
            'quoteId': quote_id,
            'timestamp': timestamp
        }
        query_string = urllib.parse.urlencode(params)
        signature = self._generate_signature(query_string)
        url = f'{self.host}/sapi/v1/convert/acceptQuote?{query_string}&signature={signature}'
        response = self._call(url, {}, method='POST')
        print('response 2', response)
