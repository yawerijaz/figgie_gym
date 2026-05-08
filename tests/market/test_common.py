import math

from figgie_gym.market.common import (
    Order,
    Price,
    Quantity,
    Side,
    Symbol,
    Trade,
    TradeSummary,
)
from tests.mocks import MockClient


def test_trade_summary_initialization() -> None:
    summary = TradeSummary.new()
    assert summary.buy_quantity == 0
    assert summary.buy_consideration == 0
    assert summary.sell_quantity == 0
    assert summary.sell_consideration == 0
    assert math.isnan(summary.average_buy_price())
    assert math.isnan(summary.average_sell_price())
    assert summary.net_quantity_change() == 0


def test_trade_summary_add_trade() -> None:
    client1 = MockClient("C1")
    client2 = MockClient("C2")
    symbol = Symbol(1)

    order1 = Order(client1, symbol, Side.BUY, Price(100), Quantity(10))
    order2 = Order(client2, symbol, Side.SELL, Price(100), Quantity(10))

    trade1 = Trade(
        client1,
        client2,
        symbol,
        Price(100),
        Quantity(10),
        order1,
        order2,
        Side.BUY,
    )

    summary = TradeSummary.new()
    summary = summary + trade1

    assert summary.buy_quantity == 10
    assert summary.buy_consideration == 1000
    assert summary.sell_quantity == 0
    assert summary.sell_consideration == 0
    assert summary.average_buy_price() == 100.0
    assert summary.net_quantity_change() == 10

    order3 = Order(client1, symbol, Side.SELL, Price(110), Quantity(5))
    order4 = Order(client2, symbol, Side.BUY, Price(110), Quantity(5))

    trade2 = Trade(
        client2,
        client1,
        symbol,
        Price(110),
        Quantity(5),
        order3,
        order4,
        Side.SELL,
    )
    summary = summary - trade2

    assert summary.buy_quantity == 10
    assert summary.buy_consideration == 1000
    assert summary.sell_quantity == 5
    assert summary.sell_consideration == 550
    assert summary.average_sell_price() == 110.0
    assert summary.net_quantity_change() == 5  # 10 - 5


def test_order_hashing() -> None:
    client = MockClient("C1")
    symbol = Symbol(1)

    order1 = Order(client, symbol, Side.BUY, Price(100))
    order2 = Order(client, symbol, Side.BUY, Price(100))

    # Different IDs, so different hash
    assert hash(order1) != hash(order2)

    # Same object, same hash
    assert hash(order1) == hash(order1)

    # Set to check uniqueness
    s = {order1, order2}
    assert len(s) == 2
