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
            'process_trending_currencies_interval',
            'percentage_difference_for_sale',
            'value_difference_for_sale',
            'currency_conversion',
            'transactions_amount',
            'total_available_amount',
            'timezone_name',
            'round_plot_numbers_to',
            'sentry_traces_sample_rate',
            'sentry_profiles_sample_rate',
        ]

        self.process_trending_currencies_interval = self.config.getint(
            'trading_frequency', 'process_trending_currencies_interval', fallback=20
        )
        self.percentage_difference_for_sale = self.config.getfloat(
            'trading_frequency', 'percentage_difference_for_sale', fallback=0.1
        )
        self.value_difference_for_sale = self.config.getint(
            'trading_frequency', 'value_difference_for_sale', fallback=100
        )

        self.currency_conversion = self.config.get(
            'trading_volume', 'currency_conversion', fallback='USDT'
        )
        self.transactions_amount = Decimal(self.config.get(
            'trading_volume', 'transactions_amount', fallback='5'
        ))
        self.total_available_amount = Decimal(self.config.get(
            'trading_volume', 'total_available_amount', fallback='100'
        ))

        self.timezone_name = self.config.get(
            'common', 'timezone_name', fallback='UTC'
        )

        self.round_plot_numbers_to = self.config.getint(
            'chart', 'round_plot_numbers_to', fallback=5
        )

        self.sentry_traces_sample_rate = self.config.getfloat(
            'sentry', 'traces_sample_rate', fallback=1.0
        )
        self.sentry_profiles_sample_rate = self.config.getfloat(
            'sentry', 'profiles_sample_rate', fallback=1.0
        )

    def change_value(self, name: str, value) -> None:
        if name not in self.parameter_list:
            raise ConfigParameterNotFound(f'Config parameter {name} not found')

        parameter = getattr(self, name)
        parameter_type = type(parameter)
        setattr(self, name, parameter_type(value))
