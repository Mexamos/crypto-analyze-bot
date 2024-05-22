from io import BytesIO
from typing import Union, Optional

import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from matplotlib.axes import Axes
from matplotlib.table import Table
from numpy import float64

from database.client import DatabaseClient
from config import Config


class GenerateChartException(Exception):
    pass


class DataForChartNotFound(GenerateChartException):
    pass


class DataForIncomesTableNotFound(GenerateChartException):
    pass


class ChartController:

    def __init__(self, db_client: DatabaseClient, config: Config) -> None:
        self.db_client = db_client
        self.config = config

    def _add_price_annotations(
        self, df: DataFrame, ax: Axes, first_price: float64, last_price: float64
    ):
        ax.annotate(
            round(first_price, self.config.round_plot_numbers_to),
            (df['date_time'][0], df['price'][0])
        )
        ax.annotate(
            round(last_price, self.config.round_plot_numbers_to),
            (df['date_time'][df.index[-1]], df['price'][df.index[-1]])
        )

    def _set_difference_label(self, ax: Axes, first_price: float64, last_price: float64):
        absolute_difference = abs(first_price - last_price)
        sum_of_values = (first_price + last_price) / 2
        difference_in_percentage = (absolute_difference / sum_of_values) * 100
        ax.set_xlabel(
            f'Difference: {round(difference_in_percentage, self.config.round_plot_numbers_to)}%', loc='left'
        )

    def _create_file_from_plot(self, artist: Union[Axes, Table]) -> BytesIO:
        fig = artist.get_figure()
        plot_file = BytesIO()
        fig.savefig(plot_file, format='png')
        plot_file.seek(0)

        plt.close('all')

        return plot_file

    def generate_chart_image(self, symbol: str) -> BytesIO:
        df: DataFrame = pd.read_sql(
            self.db_client.get_currency_price_query(symbol),
            self.db_client.get_connection(),
        )
        if len(df.values) == 0:
            raise DataForChartNotFound(f'Data for chart with symbol - {symbol} not found')

        df = df[['date_time', 'price']]
        ax: Axes = df.plot(
            x='date_time', y='price', kind='line',
            title=f'Chart of {symbol} prices', marker='o', color={"price": "#FFD700"}
        )

        first_price = df['price'][0]
        last_price = df['price'][df.index[-1]]
        self._add_price_annotations(df, ax, first_price, last_price)

        self._set_difference_label(ax, first_price, last_price)

        return self._create_file_from_plot(ax)

    def _create_table_plot(self, df: DataFrame) -> Table:
        fix, ax = plt.subplots()
        ax.axis('off')
        return pd.plotting.table(ax, df, loc='center', cellLoc='center')

    def _change_tabple_plot_sizes(self, table: Table):
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        table.auto_set_column_width((0, 1, 2, 3))

    def generate_incomes_table(self, symbol: Optional[str] = None) -> BytesIO:
        df: DataFrame = pd.read_sql(
            self.db_client.get_income_query(symbol),
            self.db_client.get_connection(),
        )
        if len(df.values) == 0:
            raise DataForIncomesTableNotFound(f'Data for incomes table with symbol - {symbol or "all"} not found')

        table = self._create_table_plot(df)
        self._change_tabple_plot_sizes(table)

        return self._create_file_from_plot(table)
