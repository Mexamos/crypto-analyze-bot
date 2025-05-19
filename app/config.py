import configparser
from decimal import Decimal


class ConfigException(Exception):
    pass


class ConfigParameterNotFound(ConfigException):
    pass


class Config:

    def __init__(self) -> None:
        self.config = configparser.RawConfigParser()
        self.config.read('./app/config.ini')

        self.parameter_list = [
            'analyze_task_interval',
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

        self.analyze_task_interval = self.config.getint(
            'trading_frequency', 'analyze_task_interval', fallback=1800
        )
        self.percentage_difference_for_sale = self.config.getfloat(
            'trading_frequency', 'percentage_difference_for_sale', fallback=0.1
        )
        self.value_difference_for_sale = self.config.getint(
            'trading_frequency', 'value_difference_for_sale', fallback=100
        )

        self.santimentapi_model_file_path = self.config.get(
            'santimentapi_model', 'model_file_path', fallback='./santimentapi_model.pkl'
        )
        self.santimentapi_scaler_file_path = self.config.get(
            'santimentapi_model', 'scaler_file_path', fallback='./santimentapi_scaler.pkl'
        )
        self.santimentapi_date_format = self.config.get(
            'santimentapi_model', 'date_format', fallback='%Y-%m-%d', raw=True
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

        self.quotes_historical_count = self.config.getint(
            'historical_data', 'quotes_historical_count', fallback=10
        )
        self.quotes_historical_interval = self.config.get(
            'historical_data', 'quotes_historical_interval', fallback='6h'
        )

        self.timezone_name = self.config.get(
            'common', 'timezone_name', fallback='UTC'
        )
        self.common_date_format = self.config.get(
            'common', 'date_format', fallback='%Y-%m-%d', raw=True
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
