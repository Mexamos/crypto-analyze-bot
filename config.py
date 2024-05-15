import configparser
from decimal import Decimal


class Config:

    def __init__(self) -> None:
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        self.request_currencies_interval = self.config.getint(
            'telegram_controller', 'request_currencies_interval', fallback=15
        )
        self.currencies_number_in_top_for_purchase = self.config.getint(
            'telegram_controller', 'currencies_number_in_top_for_purchase', fallback=5
        )
        self.percentage_difference_for_sale = self.config.getfloat(
            'telegram_controller', 'percentage_difference_for_sale', fallback=0.1
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
