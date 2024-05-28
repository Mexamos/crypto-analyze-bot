import configparser
from decimal import Decimal


class ConfigException(Exception):
    pass


class ConfigParameterNotFound(ConfigException):
    pass


class Config:

    def __init__(self) -> None:
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        self.parameter_list = [
            'process_trending_currencies_interval',
            'percentage_difference_for_sale',
            'value_difference_for_sale',
            'process_out_of_trend_currencies_interval',
            'timezone_name',
            'currency_conversion',
            'transactions_amount',
            'total_available_amount',
            'round_plot_numbers_to',
        ]

        self.process_trending_currencies_interval = self.config.getint(
            'telegram_controller', 'process_trending_currencies_interval', fallback=40
        )
        self.percentage_difference_for_sale = self.config.getfloat(
            'telegram_controller', 'percentage_difference_for_sale', fallback=0.1
        )
        self.value_difference_for_sale = self.config.getint(
            'telegram_controller', 'value_difference_for_sale', fallback=100
        )

        self.process_out_of_trend_currencies_interval = self.config.getint(
            'telegram_controller', 'process_out_of_trend_currencies_interval', fallback=20
        )

        self.timezone_name = self.config.get(
            'telegram_controller', 'timezone_name', fallback='UTC'
        )

        self.currency_conversion = self.config.get(
            'telegram_controller', 'currency_conversion', fallback='USDT'
        )
        self.transactions_amount = Decimal(self.config.get(
            'telegram_controller', 'transactions_amount', fallback='5'
        ))
        self.total_available_amount = Decimal(self.config.get(
            'telegram_controller', 'total_available_amount', fallback='100'
        ))

        self.round_plot_numbers_to = self.config.getint(
            'chart', 'round_plot_numbers_to', fallback=5
        )

    def change_value(self, name: str, value) -> None:
        if name not in self.parameter_list:
            raise ConfigParameterNotFound(f'Config parameter {name} not found')

        parameter = getattr(self, name)
        parameter_type = type(parameter)
        setattr(self, name, parameter_type(value))
