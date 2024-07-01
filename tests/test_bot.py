from datetime import datetime
from decimal import Decimal

from app.bot_controller import BotController


class TestBotController:

    async def test_me(
        self, fake_bot_controller: BotController, fake_context,
        mock_trending_gainers_losers, mock_filter_currencies_by_binance,
        mock_get_quotes_historical, mock_get_quotes_latest
    ):

        currencies = {
            "data": [
                {
                    "id": 5426,
                    "name": "Solana",
                    "symbol": "SOL",
                    "slug": "solana",
                    "num_market_pairs": 695,
                    "date_added": "2020-04-10T00:00:00.000Z",
                    "tags": [],
                    "max_supply": None,
                    "circulating_supply": 462513402.1354087,
                    "total_supply": 579229728.7978708,
                    "is_active": 1,
                    "infinite_supply": True,
                    "platform": None,
                    "cmc_rank": 5,
                    "is_fiat": 0,
                    "self_reported_circulating_supply": None,
                    "self_reported_market_cap": None,
                    "tvl_ratio": None,
                    "last_updated": "2024-06-30T13:02:00.000Z",
                    "quote": {
                        "USD": {
                            "price": 152,
                            "percent_change_24h": 0.85333811,
                        }
                    }
                },
            ]
        }

        historical = {
            "data": {
                "quotes": [
                    {
                        "timestamp": "2024-06-28T18:25:00.000Z",
                        "quote": {
                            "USD": {
                                "price": 150,
                            }
                        }
                    },
                    {
                        "timestamp": "2024-06-29T00:25:00.000Z",
                        "quote": {
                            "USD": {
                                "price": 150,
                            }
                        }
                    },
                    {
                        "timestamp": "2024-06-29T06:25:00.000Z",
                        "quote": {
                            "USD": {
                                "price": 150,
                            }
                        }
                    },
                    {
                        "timestamp": "2024-06-29T12:25:00.000Z",
                        "quote": {
                            "USD": {
                                "price": 150,
                            }
                        }
                    },
                    {
                        "timestamp": "2024-06-29T18:25:00.000Z",
                        "quote": {
                            "USD": {
                                "price": 150,
                            }
                        }
                    },
                    {
                        "timestamp": "2024-06-30T00:25:00.000Z",
                        "quote": {
                            "USD": {
                                "price": 150,
                            }
                        }
                    },
                    {
                        "timestamp": "2024-06-30T06:25:00.000Z",
                        "quote": {
                            "USD": {
                                "price": 150,
                            }
                        }
                    },
                    {
                        "timestamp": "2024-06-30T12:25:00.000Z",
                        "quote": {
                            "USD": {
                                "price": 150,
                            }
                        }
                    }
                ],
                "id": 5426,
                "name": "Solana",
                "symbol": "SOL"
            }
        }

        mock_trending_gainers_losers(currencies)
        mock_filter_currencies_by_binance(currencies['data'])

        mock_get_quotes_historical(historical)

        await fake_bot_controller.process_trending_currencies(fake_context)

        print(' ')

        historical["data"]["quotes"].append(
            {
                "timestamp": "2024-06-28T18:25:00.000Z",
                "quote": {
                    "USD": {
                        "price": currencies['data'][0]['quote']['USD']['price'],
                    }
                }
            }
        )
        mock_get_quotes_historical(historical)

        currency = {
            "data": {
                "5426": {
                    "id": 5426,
                    "name": "Solana",
                    "symbol": "SOL",
                    "slug": "solana",
                    "num_market_pairs": 696,
                    "date_added": "2020-04-10T00:00:00.000Z",
                    "tags": [],
                    "max_supply": None,
                    "circulating_supply": 462666118.12005615,
                    "total_supply": 579223820.4284009,
                    "is_active": 1,
                    "infinite_supply": True,
                    "platform": None,
                    "cmc_rank": 5,
                    "is_fiat": 0,
                    "self_reported_circulating_supply": None,
                    "self_reported_market_cap": None,
                    "tvl_ratio": None,
                    "last_updated": "2024-07-01T16:20:00.000Z",
                    "quote": {
                        "USD": {
                            "price": 148,
                            "volume_24h": 1887692562.6993446,
                            "volume_change_24h": 46.0252,
                            "percent_change_1h": 0.30749107,
                            "percent_change_24h": 3.79060988,
                            "percent_change_7d": 16.89213398,
                            "percent_change_30d": -11.11775113,
                            "percent_change_60d": 7.08263992,
                            "percent_change_90d": -16.88598286,
                            "market_cap": 68666178355.278076,
                            "market_cap_dominance": 2.9524,
                            "fully_diluted_market_cap": 85964985555.4,
                            "tvl": None,
                            "last_updated": "2024-07-01T16:20:00.000Z"
                        }
                    }
                }
            }
        }
        mock_get_quotes_latest(currency)

        await fake_bot_controller.process_trending_currencies(fake_context)

        # fake_bot_controller._create_currency_price(
        #     cmc_id=1,
        #     symbol='BTC',
        #     price=Decimal('30000'),
        #     date_time=datetime.now(),
        # )

        # await fake_bot_controller._create_income('BTC', Decimal('30300'))
