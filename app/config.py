import configparser
from decimal import Decimal


class ConfigException(Exception):
    pass


class ConfigParameterNotFound(ConfigException):
    pass


class Config:

    def __init__(self) -> None:
        self.config = configparser.ConfigParser()
        self.config.read('./app/config.ini')

        self.parameter_list = [
            'base_asset',
            'trade_asset',
            'symbol',
            'minimum_trade_amount',
            'position',
            'window_high',
            'window_medium',
            'window_low',
            'vol_window',
            'sentry_traces_sample_rate',
            'sentry_profiles_sample_rate',
        ]

        self.base_asset = self.config.get(
            'trading', 'base_asset', fallback='USDT'
        )
        self.trade_asset = self.config.get(
            'trading', 'trade_asset'
        )
        self.symbol = f'{self.trade_asset}{self.base_asset}'
        self.minimum_trade_amount = self.config.getint(
            'trading', 'minimum_trade_amount', fallback=10
        )
        self.position = self.config.get(
            'trading', 'position', fallback=None
        )

        self.window_high = self.config.getint(
            'trading', 'window_high', fallback=30
        )
        self.window_medium = self.config.getint(
            'trading', 'window_medium', fallback=5
        )
        self.window_low = self.config.getint(
            'trading', 'window_low', fallback=5
        )
        self.vol_window = self.config.getint(
            'trading', 'vol_window', fallback=30
        )
        self.adx_threshold = self.config.getint(
            'trading', 'adx_threshold', fallback=20
        )
        self.atr_period = self.config.getint(
            'trading', 'atr_period', fallback=14
        )
        self.atr_stop_multiplier = self.config.getfloat(
            'trading', 'atr_stop_multiplier', fallback=2.0
        )

        self.sentry_traces_sample_rate = self.config.getfloat(
            'sentry', 'traces_sample_rate', fallback=1.0
        )
        self.sentry_profiles_sample_rate = self.config.getfloat(
            'sentry', 'profiles_sample_rate', fallback=1.0
        )

        self.logs_file_path = self.config.get(
            'logs', 'file_path', fallback='./logs/bot.log'
        )

    def change_value(self, name: str, value) -> None:
        if name not in self.parameter_list:
            raise ConfigParameterNotFound(f'Config parameter {name} not found')

        parameter = getattr(self, name)
        parameter_type = type(parameter)
        setattr(self, name, parameter_type(value))
