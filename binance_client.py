import hmac
import hashlib
from typing import Optional

import requests

# https://binance-docs.github.io/apidocs/spot/en/#convert-endpoints

# Для аналитики использовать один из этих двух
# GET /sapi/v1/convert/exchangeInfo
# POST /sapi/v1/convert/getQuote

# POST /sapi/v1/convert/getQuote
# Send quote request
# Return quoteId

# POST /sapi/v1/convert/acceptQuote
# Accept Quote (TRADE)
# Return orderId

# GET /sapi/v1/convert/orderStatus
# Order status (USER_DATA)
# Check order status

class BinanceException(Exception):
    pass


class BinanceClient:

    def __init__(self, api_key: str, secret_key: str) -> None:
        self.host = 'https://api.binance.com'
        self.api_key = api_key
        self.secret_key = secret_key

    def find_exchange_info(self, toAsset: str, fromAsset: str):
        url = self.host + '/sapi/v1/convert/exchangeInfo'
        headers = {
            'Content-Type': 'application/json',
            'X-MBX-APIKEY': self.api_key
        }
        params = {
            'fromAsset': fromAsset,
            'toAsset': toAsset,
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            return response.json()
        except BaseException as ex:
            raise BinanceException(str(ex))


# # query_string = 'fromAsset=USDT&toAsset=SOL&fromAmount=5'
# url = 'https://api.binance.com/sapi/v1/convert/getQuote?fromAsset=USDT&toAsset=SOL&fromAmount=5&timestamp=1712336207023&signature=<replace_me>'
# headers = {
#     'Content-Type': 'application/json',
#     'X-MBX-APIKEY': BINANCE_API_KEY
# }

# # m = hmac.new(BINANCE_SECRET_KEY.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256)
# # signature = m.hexdigest()
# # print('signature', signature)

# response = requests.post(url, headers=headers)
# print('response', response.status_code, response.json())