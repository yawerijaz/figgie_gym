import pytest

from figgie_gym.market.common import Order, Price, Side, Symbol
from figgie_gym.market.exception import UnmatchableTradeError
from figgie_gym.market.validate import validate_matchable
from tests.mocks import MockClient


def test_validate_matchable_success() -> None:
    client1 = MockClient("C1")
    client2 = MockClient("C2")
    symbol = Symbol(1)

    # Buy matches Sell at same price
    buy_order = Order(client1, symbol, Side.BUY, Price(100))
    sell_order = Order(client2, symbol, Side.SELL, Price(100))
    validate_matchable(buy_order, sell_order)

    # Sell matches Buy at same price
    validate_matchable(sell_order, buy_order)

    # Buy matches Sell at lower price (price improvement)
    # Hitting: Buy @ 101, Standing: Sell @ 100 -> Valid
    buy_aggressive = Order(client1, symbol, Side.BUY, Price(101))
    validate_matchable(buy_aggressive, sell_order)

    # Sell matches Buy at higher price (price improvement)
    # Hitting: Sell @ 99, Standing: Buy @ 100 -> Valid
    sell_aggressive = Order(client2, symbol, Side.SELL, Price(99))
    validate_matchable(sell_aggressive, buy_order)


def test_validate_matchable_failures() -> None:
    client1 = MockClient("C1")
    client2 = MockClient("C2")
    symbol = Symbol(1)

    # Same side
    buy_order1 = Order(client1, symbol, Side.BUY, Price(100))
    buy_order2 = Order(client2, symbol, Side.BUY, Price(100))
    with pytest.raises(UnmatchableTradeError):
        validate_matchable(buy_order1, buy_order2)

    # Different symbol
    symbol2 = Symbol(2)
    sell_order_diff_sym = Order(client2, symbol2, Side.SELL, Price(100))
    with pytest.raises(UnmatchableTradeError):
        validate_matchable(buy_order1, sell_order_diff_sym)

    # Self match
    sell_order_self = Order(client1, symbol, Side.SELL, Price(100))
    with pytest.raises(UnmatchableTradeError):
        validate_matchable(buy_order1, sell_order_self)

    # Price mismatch (Less aggressive)
    # Hitting: Buy @ 99, Standing: Sell @ 100 -> Invalid
    buy_low = Order(client1, symbol, Side.BUY, Price(99))
    sell_high = Order(client2, symbol, Side.SELL, Price(100))
    with pytest.raises(
        UnmatchableTradeError,
    ):
        validate_matchable(buy_low, sell_high)

    # Hitting: Sell @ 101, Standing: Buy @ 100 -> Invalid
    sell_high_hit = Order(client2, symbol, Side.SELL, Price(101))
    buy_low_stand = Order(client1, symbol, Side.BUY, Price(100))
    with pytest.raises(
        UnmatchableTradeError,
    ):
        validate_matchable(sell_high_hit, buy_low_stand)
