import pytest

from figgie_gym.market.common import (
    Order,
    Price,
    Quantity,
    Side,
    Symbol,
)
from figgie_gym.market.exception import InvalidOrderStatusError
from figgie_gym.market.market import Market


def test_match_buy_sell() -> None:
    market = Market([Symbol(1)])
    client1 = market.new_client("C1")
    client2 = market.new_client("C2")
    ticker = market.create_trade_ticker_tape()

    # Sell order
    market.process_order(Order(client1, Symbol(1), Side.SELL, Price(100)))

    # Buy order matches
    market.process_order(Order(client2, Symbol(1), Side.BUY, Price(100)))

    assert len(ticker.trades) == 1
    assert ticker.trades[0].price == 100
    assert ticker.trades[0].buyer == client2
    assert ticker.trades[0].seller == client1


def test_match_better_price() -> None:
    market = Market([Symbol(1)])
    client1 = market.new_client("C1")
    client2 = market.new_client("C2")
    ticker = market.create_trade_ticker_tape()

    # Sell order at 100
    market.process_order(Order(client1, Symbol(1), Side.SELL, Price(100)))

    # Buy order at 101 matches at 100 (best available price)
    market.process_order(Order(client2, Symbol(1), Side.BUY, Price(101)))

    assert len(ticker.trades) == 1
    assert ticker.trades[0].price == 100


def test_no_match() -> None:
    market = Market([Symbol(1)])
    client1 = market.new_client("C1")
    client2 = market.new_client("C2")
    ticker = market.create_trade_ticker_tape()

    # Sell order at 100
    market.process_order(Order(client1, Symbol(1), Side.SELL, Price(100)))

    # Buy order at 99 does not match
    market.process_order(Order(client2, Symbol(1), Side.BUY, Price(99)))

    assert len(ticker.trades) == 0
    assert len(ticker.trades) == 0


def test_time_priority() -> None:
    market = Market([Symbol(1)])
    client1 = market.new_client("C1")
    client2 = market.new_client("C2")
    client3 = market.new_client("C3")
    ticker = market.create_trade_ticker_tape()

    # Two sell orders at 100
    market.process_order(Order(client1, Symbol(1), Side.SELL, Price(100)))
    market.process_order(Order(client2, Symbol(1), Side.SELL, Price(100)))

    # Buy order matches the first sell order (C1)
    market.process_order(Order(client3, Symbol(1), Side.BUY, Price(100)))

    assert len(ticker.trades) == 1

    trade = ticker.trades[0]
    assert trade.seller == client1
    assert trade.buyer == client3

    # Client 2 sees the same trade
    assert ticker.trades[0] == trade


def test_cancel_order() -> None:
    market = Market([Symbol(1)])
    client1 = market.new_client("C1")
    client2 = market.new_client("C2")
    ticker = market.create_trade_ticker_tape()

    # Sell order
    order = Order(client1, Symbol(1), Side.SELL, Price(100))
    market.process_order(order)

    # Cancel it. We need to access the internal queue to cancel properly
    # since the public API doesn't expose cancel easily in this simple version.
    # However, looking at the code, `ActiveOrderQueued.cancel` exists but isn't exposed via Market.
    # For this test, we'll inspect the book to find the order and cancel it.
    book = market.books[Symbol(1)]
    queue = book.asks.limits[Price(100)]
    active_order = queue.first
    active_order.cancel()

    # Buy order should not match
    market.process_order(Order(client2, Symbol(1), Side.BUY, Price(100)))

    assert len(ticker.trades) == 0


def test_cancel_non_active_order() -> None:
    market = Market([Symbol(1)])
    client1 = market.new_client("C1")
    client2 = market.new_client("C2")
    ticker = market.create_trade_ticker_tape()

    # Sell order
    order = Order(client1, Symbol(1), Side.SELL, Price(100))
    market.process_order(order)

    # Cancel it. We need to access the internal queue to cancel properly
    # since the public API doesn't expose cancel easily in this simple version.
    # However, looking at the code, `ActiveOrderQueued.cancel` exists but isn't exposed via Market.
    # For this test, we'll inspect the book to find the order and cancel it.
    book = market.books[Symbol(1)]
    queue = book.asks.limits[Price(100)]
    active_order = queue.first
    active_order.cancel()

    # Illegally cancel a cancelled order
    with pytest.raises(InvalidOrderStatusError):
        active_order.cancel()

    # Buy order should not match
    market.process_order(Order(client2, Symbol(1), Side.BUY, Price(100)))
    active_order2 = book.bids.limits[Price(100)].first

    # Match the order now
    order2 = Order(client1, Symbol(1), Side.SELL, Price(100))
    market.process_order(order2)

    # Illegally cancel a filled order
    with pytest.raises(InvalidOrderStatusError):
        active_order2.cancel()

    assert len(ticker.trades) == 1


def test_ladder() -> None:
    market = Market([Symbol(1)])
    client1 = market.new_client("C1")

    market.process_order(Order(client1, Symbol(1), Side.BUY, Price(100)))
    market.process_order(Order(client1, Symbol(1), Side.SELL, Price(101)))

    book = market.books[Symbol(1)]
    ladder = book.ladder()

    assert ladder[Price(100)] == (Quantity(1), None)
    assert ladder[Price(101)] == (None, Quantity(1))
