import sentry_sdk

from app.config import Config


class SentryClient:

    def __init__(self, dsn: str, config: Config) -> None:
        sentry_sdk.init(
            dsn=dsn,
            traces_sample_rate=config.sentry_traces_sample_rate,
            profiles_sample_rate=config.sentry_profiles_sample_rate,
        )

    def capture_exception(self, exception: BaseException):
        sentry_sdk.capture_exception(error=exception)
