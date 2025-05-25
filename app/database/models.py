from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, String, DECIMAL, Column, Integer, Numeric
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class CurrencyPrice(Base):
    __tablename__ = "currency_price"

    id: Mapped[int] = mapped_column(primary_key=True)
    cmc_id: Mapped[int]
    symbol: Mapped[str] = mapped_column(String(10))
    price: Mapped[Decimal] = mapped_column(DECIMAL(50, 30))
    date_time: Mapped[datetime] = mapped_column(DateTime)


class Income(Base):
    __tablename__ = "income"

    id: Mapped[int] = mapped_column(primary_key=True)
    cmc_id: Mapped[int]
    symbol: Mapped[str] = mapped_column(String(10))
    value: Mapped[Decimal] = mapped_column(DECIMAL(50, 30))
    date_time: Mapped[datetime] = mapped_column(DateTime)


class CryptocurrencyPurchase(Base):
    __tablename__ = 'cryptocurrency_purchase'

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    name = Column(String, nullable=False)
    coingecko_id = Column(String, nullable=True)
    price = Column(Numeric, nullable=False)
    date = Column(DateTime, nullable=False)
