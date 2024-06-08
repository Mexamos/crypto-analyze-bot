from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, String, DECIMAL
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
    symbol: Mapped[str] = mapped_column(String(10))
    value: Mapped[Decimal] = mapped_column(DECIMAL(50, 30))
    date_time: Mapped[datetime] = mapped_column(DateTime)
