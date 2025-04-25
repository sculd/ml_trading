import os
from binance.client import Client

_API_KEY = os.getenv('API_KEY_BINANCE')
_API_SECRET = os.getenv('API_SECRET_BINANCE')


_client = None


def get_client():
    global _client
    if _client is not None:
        return _client
    _client = Client(_API_KEY, _API_SECRET)
    return _client

