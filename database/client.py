from typing import List, Optional

from sqlalchemy import create_engine, select, delete
from sqlalchemy.orm import Session

from database.models import Base, CurrencyPrice, Income


class DatabaseClient:

    def __init__(self) -> None:
        self.engine = create_engine("sqlite://", echo=False)
        Base.metadata.create_all(self.engine)

    def get_connection(self):
        return self.engine.connect()

    def get_currency_price_query(self, symbol: str):
        return select(CurrencyPrice).where(CurrencyPrice.symbol == symbol)

    def find_currency_price_by_symbol(self, symbol: str) -> List[CurrencyPrice]:
        session = Session(self.engine)
        stmt = self.get_currency_price_query(symbol)
        return session.scalars(stmt).all()

    def find_currency_price_symbols(self) -> List[str]:
        session = Session(self.engine)
        stmt = select(CurrencyPrice.symbol).distinct()
        return session.scalars(stmt).all()

    def create_currency_price(
        self, symbol: str, price, percent_change_24h, date_time
    ):
        with Session(self.engine) as session:
            currency = CurrencyPrice(
                symbol=symbol,
                price=price,
                percent_change_24h=percent_change_24h,
                date_time=date_time,
            )
            session.add(currency)
            session.commit()

    def delete_currency_prices(self, symbol: str):
        with Session(self.engine) as session:
            stmt = delete(CurrencyPrice).where(CurrencyPrice.symbol == symbol)
            session.execute(stmt)
            session.commit()

    def get_income_query(self, symbol: Optional[str] = None):
        select_query = select(Income)
        if symbol:
            select_query = select_query.where(Income.symbol == symbol)
        return select_query

    def find_income_symbols(self) -> List[str]:
        session = Session(self.engine)
        stmt = select(Income.symbol).distinct()
        return session.scalars(stmt).all()

    def create_income(
        self, symbol: str, value, date_time
    ):
        with Session(self.engine) as session:
            income = Income(
                symbol=symbol,
                value=value,
                date_time=date_time,
            )
            session.add(income)
            session.commit()
