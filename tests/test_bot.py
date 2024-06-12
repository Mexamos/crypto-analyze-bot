from datetime import datetime
from decimal import Decimal

from app.bot_controller import BotController


class TestBotController:

    async def test_me(self, fake_bot_controller: BotController):
        fake_bot_controller._create_currency_price(
            cmc_id=1,
            symbol='BTC',
            price=Decimal('30000'),
            date_time=datetime.now(),
        )

        await fake_bot_controller._create_income('BTC', Decimal('30300'))
