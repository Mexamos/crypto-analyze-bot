from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from sqlalchemy import create_engine, select, delete, text, func
from sqlalchemy.orm import Session

from app.database.models import Base, CurrencyPrice, Income, CryptocurrencyPurchase


class DatabaseClient:

    def __init__(self) -> None:
        # in-memory database
        self.engine = create_engine("sqlite://")
        Base.metadata.create_all(self.engine)

    def get_connection(self):
        return self.engine.connect()

    def count_all_currency_price(self) -> int:
        session = Session(self.engine)
        result = session.query(func.count(CurrencyPrice.id)).scalar()
        session.close()
        return result

    def get_currency_price_query(self, symbol: str):
        return select(CurrencyPrice).where(CurrencyPrice.symbol == symbol)

    def find_currency_price_by_symbol(self, symbol: str) -> List[CurrencyPrice]:
        session = Session(self.engine)
        stmt = self.get_currency_price_query(symbol)
        result = session.scalars(stmt).all()
        session.close()
        return result

    def find_currency_price_symbols(self) -> List[str]:
        session = Session(self.engine)
        stmt = select(CurrencyPrice.symbol).distinct()
        result = session.scalars(stmt).all()
        session.close()
        return result
    
    def find_currency_price_by_cmc_id(self, cmc_id: int) -> List[CurrencyPrice]:
        session = Session(self.engine)
        stmt = select(CurrencyPrice).where(CurrencyPrice.cmc_id == cmc_id)
        result = session.scalars(stmt).all()
        session.close()
        return result

    def find_currency_price_cmc_ids(self) -> List[str]:
        session = Session(self.engine)
        stmt = select(CurrencyPrice.cmc_id).distinct()
        result = session.scalars(stmt).all()
        session.close()
        return result

    def find_first_currency_prices_grouped_by_cmc_id(self) -> List[CurrencyPrice]:
        session = Session(self.engine)
        query = text('''
            SELECT id, cmc_id, symbol, price, MIN(date_time) 
            FROM currency_price GROUP BY cmc_id ORDER BY id;
        ''')
        result = session.execute(query).all()
        session.close()
        return result

    def create_currency_price(
        self, cmc_id: int, symbol: str, price: Decimal, date_time: datetime
    ):
        with Session(self.engine) as session:
            currency = CurrencyPrice(
                cmc_id=cmc_id,
                symbol=symbol,
                price=price,
                date_time=date_time,
            )
            session.add(currency)
            session.commit()

    def delete_currency_prices(self, cmc_id: int):
        with Session(self.engine) as session:
            stmt = delete(CurrencyPrice).where(CurrencyPrice.cmc_id == cmc_id)
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
        result = session.scalars(stmt).all()
        session.close()
        return result

    def find_income_sum_by_cmc_id(self) -> List[Income]:
        session = Session(self.engine)
        query = text('''
            SELECT cmc_id, sum(value)
            FROM income GROUP BY cmc_id;                     
        ''')
        result = session.execute(query).all()
        session.close()
        return result

    def create_income(
        self, cmc_id: int, symbol: str, value, date_time
    ):
        with Session(self.engine) as session:
            income = Income(
                cmc_id=cmc_id,
                symbol=symbol,
                value=value,
                date_time=date_time,
            )
            session.add(income)
            session.commit()

    def find_cryptocurrency_purchase_by_symbol(self, symbol: str) -> Optional[CryptocurrencyPurchase]:
        with Session(self.engine) as session:
            stmt = select(CryptocurrencyPurchase).where(CryptocurrencyPurchase.symbol == symbol)
            result = session.scalars(stmt).first()
            session.close()
            return result

    def create_cryptocurrency_purchase(
        self, symbol: str, name: str, price: Decimal, date: datetime, coingecko_id: Optional[str] = None
    ):
        with Session(self.engine) as session:
            purchase = CryptocurrencyPurchase(
                symbol=symbol,
                name=name,
                price=price,
                date=date,
                coingecko_id=coingecko_id
            )
            session.add(purchase)
            session.commit()
