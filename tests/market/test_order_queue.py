import pytest

from figgie_gym.market.common import (
    Client,
    Order,
    Price,
    Quantity,
    Side,
    Symbol,
)
from figgie_gym.market.exception import (
    InvalidOrderStatusError,
    QueueEmptyError,
    UnmatchableTradeError,
)
from figgie_gym.market.order_fill import OrderInMarket
from figgie_gym.market.order_queue import PriceLimitQueue
from tests.mocks import MockQueueOwner


def test_queue_initialization(
    client1: Client,
    symbol: Symbol,
    queue_owner: MockQueueOwner,
) -> None:
    order = Order(client1, symbol, Side.BUY, Price(100))
    queue = PriceLimitQueue(queue_owner, order.price)
    queue.enqueue(OrderInMarket.new(order))

    assert queue.price == 100
    assert queue.total_quantity == 1
    assert queue.first.order == order
    assert queue.last.order == order


def test_queue_add_append(
    client1: Client,
    client2: Client,
    symbol: Symbol,
    queue_owner: MockQueueOwner,
) -> None:
    order1 = Order(client1, symbol, Side.BUY, Price(100))
    queue = PriceLimitQueue(queue_owner, order1.price)
    queue.enqueue(OrderInMarket.new(order1))

    order2 = Order(client2, symbol, Side.BUY, Price(100))
    queue.enqueue(OrderInMarket.new(order2))

    assert queue.total_quantity == 2
    assert queue.first.order == order1
    assert queue.last.order == order2
    assert queue.first.next == queue.last
    assert queue.last.prev == queue.first


def test_queue_match(
    client1: Client,
    client2: Client,
    symbol: Symbol,
    queue_owner: MockQueueOwner,
) -> None:
    order1 = Order(client1, symbol, Side.BUY, Price(100))
    queue = PriceLimitQueue(queue_owner, order1.price)
    queue.enqueue(OrderInMarket.new(order1))

    # Matching sell order
    sell_order = Order(client2, symbol, Side.SELL, Price(100))
    trades, order_status_updates = queue.match(OrderInMarket.new(sell_order))

    assert len(trades) == 1
    assert trades[0].price == 100
    assert trades[0].buyer == client1
    assert trades[0].seller == client2
    assert queue.total_quantity == 0
    # Check if prune was called
    assert queue_owner.pruned_prices == [Price(100)]
    assert queue_owner.pruned_orders == [order1]

    assert len(order_status_updates) == 2
    assert (
        order_status_updates[0].cumulative_quantity
        == order_status_updates[1].cumulative_quantity
        == 1
    )
    assert (
        order_status_updates[0].remaining_quantity
        == order_status_updates[1].remaining_quantity
        == 0
    )
    assert (
        order_status_updates[0].cumulative_consideration
        == order_status_updates[1].cumulative_consideration
        == 100
    )
    assert {order_status_updates[0].order, order_status_updates[1].order} == {
        order1,
        sell_order,
    }


def test_queue_match_multiple(
    client1: Client,
    client2: Client,
    client3: Client,
    symbol: Symbol,
    queue_owner: MockQueueOwner,
) -> None:
    order1 = Order(client1, symbol, Side.BUY, Price(100))
    queue = PriceLimitQueue(queue_owner, order1.price)
    queue.enqueue(OrderInMarket.new(order1))

    order2 = Order(client3, symbol, Side.BUY, Price(100))
    queue.enqueue(OrderInMarket.new(order2))

    # Match first order
    sell_order1 = Order(client2, symbol, Side.SELL, Price(100))
    trades1, order_status_updates1 = queue.match(OrderInMarket.new(sell_order1))

    assert len(trades1) == 1
    assert trades1[0].buyer == client1
    assert queue.total_quantity == 1
    assert queue.first.order == order2

    assert len(order_status_updates1) == 2
    assert (
        order_status_updates1[0].cumulative_quantity
        == order_status_updates1[1].cumulative_quantity
        == 1
    )
    assert (
        order_status_updates1[0].remaining_quantity
        == order_status_updates1[1].remaining_quantity
        == 0
    )
    assert (
        order_status_updates1[0].cumulative_consideration
        == order_status_updates1[1].cumulative_consideration
        == 100
    )
    assert {order_status_updates1[0].order, order_status_updates1[1].order} == {
        order1,
        sell_order1,
    }

    # Match second order
    sell_order2 = Order(client2, symbol, Side.SELL, Price(100))
    trades2, order_status_updates2 = queue.match(OrderInMarket.new(sell_order2))

    assert len(trades2) == 1
    assert trades2[0].buyer == client3
    assert queue.total_quantity == 0
    assert queue_owner.pruned_prices == [Price(100)]
    assert queue_owner.pruned_orders == [order1, order2]

    assert len(order_status_updates2) == 2
    assert (
        order_status_updates2[0].cumulative_quantity
        == order_status_updates2[1].cumulative_quantity
        == 1
    )
    assert (
        order_status_updates2[0].remaining_quantity
        == order_status_updates2[1].remaining_quantity
        == 0
    )
    assert (
        order_status_updates2[0].cumulative_consideration
        == order_status_updates2[1].cumulative_consideration
        == 100
    )
    assert {order_status_updates2[0].order, order_status_updates2[1].order} == {
        order2,
        sell_order2,
    }


def test_queue_validation_errors(
    client1: Client,
    client2: Client,
    symbol: Symbol,
    queue_owner: MockQueueOwner,
) -> None:
    buy_order = Order(client1, symbol, Side.BUY, Price(100))
    queue = PriceLimitQueue(queue_owner, buy_order.price)
    queue.enqueue(OrderInMarket.new(buy_order))

    # Same side
    with pytest.raises(UnmatchableTradeError):
        queue.match(
            OrderInMarket.new(Order(client2, symbol, Side.BUY, Price(100))),
        )

    # Self match
    with pytest.raises(UnmatchableTradeError):
        queue.match(
            OrderInMarket.new(Order(client1, symbol, Side.SELL, Price(100))),
        )

    # Symbol mismatch
    with pytest.raises(UnmatchableTradeError):
        queue.match(
            OrderInMarket.new(Order(client2, Symbol(2), Side.SELL, Price(100))),
        )

    # Price mismatch (Sell price > Buy price)
    # Queue has Buy @ 100. Incoming Sell @ 101. Should fail.
    with pytest.raises(UnmatchableTradeError):
        queue.match(
            OrderInMarket.new(Order(client2, symbol, Side.SELL, Price(101))),
        )


def test_queue_cancel_order(
    client1: Client,
    symbol: Symbol,
    queue_owner: MockQueueOwner,
) -> None:
    order = Order(client1, symbol, Side.BUY, Price(100))
    queue = PriceLimitQueue(queue_owner, order.price)
    queue.enqueue(OrderInMarket.new(order))

    active_order = queue.first
    assert active_order.status == "active"
    assert queue.total_quantity == 1

    active_order.cancel()

    assert active_order.status == "canceled"
    assert queue.total_quantity == 0
    assert queue_owner.pruned_prices == [Price(100)]


def test_invalid_order_status(
    client1: Client,
    symbol: Symbol,
    queue_owner: MockQueueOwner,
) -> None:
    order = Order(client1, symbol, Side.BUY, Price(100))
    queue = PriceLimitQueue(queue_owner, order.price)
    queue.enqueue(OrderInMarket.new(order))

    active_order = queue.first
    active_order.cancel()

    # Try to cancel again
    with pytest.raises(InvalidOrderStatusError):
        active_order.cancel()

    # Try to fill
    with pytest.raises(InvalidOrderStatusError):
        active_order.execute_quantity(Quantity(1), Price(100))


def test_queue_empty_error(
    client1: Client,
    client2: Client,
    symbol: Symbol,
    queue_owner: MockQueueOwner,
) -> None:
    queue = PriceLimitQueue(queue_owner, Price(100))

    with pytest.raises(QueueEmptyError):
        _ = queue.first

    with pytest.raises(QueueEmptyError):
        _ = queue.last

    order1 = Order(client1, symbol, Side.BUY, Price(100))
    queue.enqueue(OrderInMarket.new(order1))

    assert queue.first
    assert queue.last

    order2 = Order(client2, symbol, Side.SELL, Price(100))
    queue.match(OrderInMarket.new(order2))

    with pytest.raises(QueueEmptyError):
        _ = queue.first

    with pytest.raises(QueueEmptyError):
        _ = queue.last

    order3 = Order(client1, symbol, Side.BUY, Price(100))
    active_order = queue.enqueue(OrderInMarket.new(order3))

    assert queue.first
    assert queue.last

    active_order.cancel()

    with pytest.raises(QueueEmptyError):
        _ = queue.first

    with pytest.raises(QueueEmptyError):
        _ = queue.last
