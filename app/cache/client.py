from redis import Redis


class CacheClient:

    def __init__(self, cache: Redis):
        self.cache = cache

        try:
            self.cache.ping()
        except Exception as e:
            raise ConnectionError(f"Unable to connect to Redis: {e}")

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache.set(key, value)

    def delete(self, key):
        self.cache.delete(key)
