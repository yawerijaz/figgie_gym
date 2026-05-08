import pytest

from figgie_gym.market.common import (
    Client,
    Order,
    OrderStatus,
    OrderStatusUpdate,
    OrderType,
    Price,
    Quantity,
    Side,
    Symbol,
)
from figgie_gym.market.exception import (
    ImpossibleMarketSituationError,
    UnmatchableTradeError,
)
from figgie_gym.market.order_book import OrderBook
from figgie_gym.market.order_fill import OrderInMarket


def test_book_add_orders(client1: Client, symbol: Symbol) -> None:
    book = OrderBook()

    # Add Buy orders
    book.process_order(Order(client1, symbol, Side.BUY, Price(100)))
    book.process_order(Order(client1, symbol, Side.BUY, Price(99)))

    # Add Sell orders
    book.process_order(Order(client1, symbol, Side.SELL, Price(102)))
    book.process_order(Order(client1, symbol, Side.SELL, Price(103)))

    top = book.top()

    assert top.bid is not None
    assert top.bid.price == 100
    assert top.bid.quantity == 1

    assert top.ask is not None
    assert top.ask.price == 102
    assert top.ask.quantity == 1

    ladder = book.ladder()
    assert ladder[Price(100)] == (Quantity(1), None)
    assert ladder[Price(99)] == (Quantity(1), None)
    assert ladder[Price(102)] == (None, Quantity(1))
    assert ladder[Price(103)] == (None, Quantity(1))


def test_book_match_basic(
    client1: Client,
    client2: Client,
    symbol: Symbol,
) -> None:
    book = OrderBook()

    # Sell @ 100
    book.process_order(Order(client1, symbol, Side.SELL, Price(100)))

    # Buy @ 100 matches
    trades, order_status_updates = book.process_order(
        Order(client2, symbol, Side.BUY, Price(100)),
    )

    assert len(trades) == 1
    assert trades[0].price == 100
    assert trades[0].buyer == client2
    assert trades[0].seller == client1

    # Book should be empty
    assert book.top().bid is None
    assert book.top().ask is None

    assert len(order_status_updates) == 2
    assert (
        order_status_updates[0].status
        is order_status_updates[1].status
        is OrderStatus.FILLED
    )
    assert (
        order_status_updates[0].remaining_quantity
        == order_status_updates[1].remaining_quantity
        == 0
    )


def test_book_match_price_improvement(
    client1: Client,
    client2: Client,
    symbol: Symbol,
) -> None:
    book = OrderBook()

    # Sell @ 100
    book.process_order(Order(client1, symbol, Side.SELL, Price(100)))

    # Buy @ 101 matches at 100
    trades, order_status_updates = book.process_order(
        Order(client2, symbol, Side.BUY, Price(101)),
    )

    assert len(trades) == 1
    assert trades[0].price == 100  # Matches standing order price

    # Book empty
    assert book.top().bid is None
    assert book.top().ask is None

    assert len(order_status_updates) == 2
    assert (
        order_status_updates[0].status
        is order_status_updates[0].status
        is OrderStatus.FILLED
    )
    assert (
        order_status_updates[0].remaining_quantity
        == order_status_updates[0].remaining_quantity
        == 0
    )


def test_book_time_priority(
    client1: Client,
    client2: Client,
    client3: Client,
    symbol: Symbol,
) -> None:
    book = OrderBook()

    # Two sells at 100
    book.process_order(Order(client1, symbol, Side.SELL, Price(100)))
    book.process_order(Order(client2, symbol, Side.SELL, Price(100)))

    # First buy matches client1 (first in)
    trades1, order_status_updates1 = book.process_order(
        Order(client3, symbol, Side.BUY, Price(100)),
    )
    assert len(trades1) == 1
    assert trades1[0].seller == client1

    # Second buy matches client2
    trades2, order_status_updates2 = book.process_order(
        Order(client3, symbol, Side.BUY, Price(100)),
    )
    assert len(trades2) == 1
    assert trades2[0].seller == client2

    assert len(order_status_updates1) == len(order_status_updates2) == 2
    assert (
        order_status_updates1[0].status
        is order_status_updates1[1].status
        is OrderStatus.FILLED
    )
    assert (
        order_status_updates2[0].remaining_quantity
        == order_status_updates2[1].remaining_quantity
        == 0
    )


def test_book_validation_errors(
    client1: Client,
    client2: Client,
    symbol: Symbol,
) -> None:
    book = OrderBook()

    # Self match
    book.process_order(Order(client1, symbol, Side.SELL, Price(100)))
    with pytest.raises(UnmatchableTradeError):
        book.process_order(Order(client1, symbol, Side.BUY, Price(100)))

    # Symbol mismatch
    book = OrderBook()
    book.process_order(Order(client1, symbol, Side.SELL, Price(100)))

    with pytest.raises(UnmatchableTradeError):
        book.process_order(Order(client2, Symbol(2), Side.BUY, Price(100)))


def test_book_cancel_order(client1: Client, symbol: Symbol) -> None:
    book = OrderBook()
    order = Order(client1, symbol, Side.BUY, Price(100))
    book.process_order(order)

    # Verify order is in book
    top_bid = book.top().bid
    assert top_bid
    assert top_bid.quantity == 1

    # Cancel order
    assert isinstance(book.cancel_order(order), OrderStatusUpdate)

    # Verify order is removed
    assert book.top().bid is None
    assert Price(100) not in book.ladder()


def test_book_cancel_non_existent_order(
    client1: Client,
    symbol: Symbol,
) -> None:
    book = OrderBook()
    order = Order(client1, symbol, Side.BUY, Price(100))

    # Try to cancel order that was never added
    assert book.cancel_order(order) is None

    # Add order and cancel it, then try to cancel again
    book.process_order(order)
    assert book.cancel_order(order) is not None
    assert book.cancel_order(order) is None


def test_impossible_market_situation(client1: Client, symbol: Symbol) -> None:
    book = OrderBook()

    # We need to manually create a crossed market because process_order matches
    # Create Buy @ 100
    buy_order = Order(client1, symbol, Side.BUY, Price(100))
    book.bids.insert(OrderInMarket.new(buy_order))

    # Create Sell @ 99 (Crossed!)
    sell_order = Order(client1, symbol, Side.SELL, Price(99))
    book.asks.insert(OrderInMarket.new(sell_order))

    with pytest.raises(ImpossibleMarketSituationError):
        book.top()


def test_ioc_order_full_fill(
    client1: Client,
    client2: Client,
    symbol: Symbol,
) -> None:
    book = OrderBook()

    # Setup: Sell @ 100 (Qty 1)
    sell_order = Order(client1, symbol, Side.SELL, Price(100))
    book.process_order(sell_order)

    # IOC Buy @ 100 (Qty 1)
    buy_order = Order(
        client2,
        symbol,
        Side.BUY,
        Price(100),
        order_type=OrderType.IMMEDIATE_OR_CANCEL,
    )

    trades, order_status_updates = book.process_order(buy_order)

    assert len(trades) == 1
    assert trades[0].quantity == 1
    assert book.asks.active_orders == {}
    assert book.bids.active_orders == {}

    assert len(order_status_updates) == 2
    assert (
        order_status_updates[0].status
        is order_status_updates[1].status
        is OrderStatus.FILLED
    )
    assert (
        order_status_updates[0].remaining_quantity
        == order_status_updates[1].remaining_quantity
        == 0
    )


def test_ioc_order_partial_fill(
    client1: Client,
    client2: Client,
    symbol: Symbol,
) -> None:
    book = OrderBook()

    # Setup: Sell @ 100 (Qty 1)
    sell_order = Order(client1, symbol, Side.SELL, Price(100))
    book.process_order(sell_order)

    # IOC Buy @ 100 (Qty 2)
    buy_order = Order(
        client2,
        symbol,
        Side.BUY,
        Price(100),
        quantity=Quantity(2),
        order_type=OrderType.IMMEDIATE_OR_CANCEL,
    )

    trades, order_status_updates = book.process_order(buy_order)

    assert len(trades) == 1
    assert trades[0].quantity == 1

    # Remaining 1 unit should NOT be in book
    assert book.bids.active_orders == {}
    assert book.asks.active_orders == {}

    assert len(order_status_updates) == 3
    assert order_status_updates[0].status is OrderStatus.FILLED
    assert order_status_updates[1].status is OrderStatus.NEW
    assert order_status_updates[2].status is OrderStatus.DEAD
    assert (
        order_status_updates[0].remaining_quantity
        == order_status_updates[2].remaining_quantity
        == 0
    )
    assert (
        order_status_updates[0].cumulative_quantity
        == order_status_updates[1].cumulative_quantity
        == order_status_updates[2].cumulative_quantity
        == 1
    )


def test_ioc_order_no_fill(
    client1: Client,
    client2: Client,
    symbol: Symbol,
) -> None:
    book = OrderBook()

    # Setup: Sell @ 101 (Qty 1)
    sell_order = Order(client1, symbol, Side.SELL, Price(101))
    book.process_order(sell_order)

    # IOC Buy @ 100 -> No match
    buy_order = Order(
        client2,
        symbol,
        Side.BUY,
        Price(100),
        order_type=OrderType.IMMEDIATE_OR_CANCEL,
    )

    trades, order_status_updates = book.process_order(buy_order)

    assert len(trades) == 0

    # Order should NOT be in book
    assert book.bids.active_orders == {}
    # Sell order remains
    assert len(book.asks.active_orders) == 1

    assert len(order_status_updates) == 1
    assert order_status_updates[0].status is OrderStatus.DEAD
    assert order_status_updates[0].remaining_quantity == 0
