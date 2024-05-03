from datetime import datetime
from typing import List
from typing import Optional

from sqlalchemy import DateTime, String
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship


class Base(DeclarativeBase):
    pass


class Currency(Base):
    __tablename__ = "currency"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(10))
    price: Mapped[str] = mapped_column(String(50))
    percent_change_24h: Mapped[str] = mapped_column(String(50))
    timestamp: Mapped[datetime] = mapped_column(DateTime)


# symbol
# quote.USD.price
# quote.USD.percent_change_24h
# текущий timestamp