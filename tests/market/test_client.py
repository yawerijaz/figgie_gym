import pytest

from figgie_gym.market.common import (
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
    OrderOperationNotAllowedError,
    PotentialSelfMatchError,
    PotentialShortSellError,
    ShortSellError,
)
from figgie_gym.market.market import Market
from figgie_gym.market.order_queue import ActiveOrderQueued

symbol = Symbol("1")


def test_client_position_matches_summarizer() -> None:
    market = Market([symbol])
    summarizer = market.create_trade_summarizer()

    # Create clients with initial positions
    client1 = market.new_client("C1", {symbol: Quantity(10)})
    client2 = market.new_client("C2", {symbol: Quantity(10)})

    # Trade: C1 buys 5 from C2
    for _ in range(5):
        client2.send_order(symbol, Side.SELL, Price(100))
        client1.send_order(symbol, Side.BUY, Price(100))

    # Check positions
    assert client1.order_managers[symbol].quantity == 15
    assert client2.order_managers[symbol].quantity == 5

    # Check summarizer
    assert summarizer.summary[symbol][client1].net_quantity_change() == 5
    assert summarizer.summary[symbol][client2].net_quantity_change() == -5

    # Verify consistency
    assert (
        client1.order_managers[symbol].quantity
        == 10 + summarizer.summary[symbol][client1].net_quantity_change()
    )
    assert (
        client2.order_managers[symbol].quantity
        == 10 + summarizer.summary[symbol][client2].net_quantity_change()
    )

    assert summarizer.last_prices[symbol] == 100


def test_short_sell_prevention_pending_orders() -> None:
    market = Market([symbol])

    # Create a client with 5 units
    client = market.new_client("C1", {symbol: Quantity(5)})

    # Place 5 sell orders (limit orders that won't match immediately)
    # We need a price that won't match. Since book is empty, any sell is fine.
    for _ in range(5):
        client.send_order(symbol, Side.SELL, Price(100))

    # Try to place 6th sell order and risk short-selling
    with pytest.raises(PotentialShortSellError):
        client.send_order(symbol, Side.SELL, Price(100))


def test_no_short_sell_prevention_pending_orders_if_ioc() -> None:
    market = Market([symbol])

    # Create a client with 5 units
    client = market.new_client("C1", {symbol: Quantity(5)})

    # Place 5 sell orders (limit orders that won't match immediately)
    # We need a price that won't match. Since book is empty, any sell is fine.
    for _ in range(4):
        client.send_order(symbol, Side.SELL, Price(100))

    # Place an IOC order
    client.send_order(
        symbol,
        Side.SELL,
        Price(100),
        order_type=OrderType.IMMEDIATE_OR_CANCEL,
    )

    # Place 6th sell order for the remaining one but no risk short-selling
    client.send_order(symbol, Side.SELL, Price(100))

    # Try to place 7th sell order and risk short-selling
    with pytest.raises(PotentialShortSellError):
        client.send_order(symbol, Side.SELL, Price(100))


def test_short_sell_prevention_cancellation() -> None:
    market = Market([symbol])

    # Create a client with 5 units
    client = market.new_client("C1", {symbol: Quantity(5)})

    # Place 5 sell orders
    for _ in range(5):
        client.send_order(symbol, Side.SELL, Price(100))

    # Get orders from book
    book = market.books[symbol]
    queue = book.asks.limits[Price(100)]

    # Cancel one order
    order_to_cancel = queue.first.order
    client.cancel_order(order_to_cancel)

    # Now we should be able to place another sell order
    client.send_order(symbol, Side.SELL, Price(100))

    # Verify we have 5 orders in book again (4 original + 1 new)
    assert queue.total_quantity == 5


def test_client_cash_and_averages_consistency() -> None:
    market = Market([symbol])
    summarizer = market.create_trade_summarizer()

    # Create clients with initial cash
    client1 = market.new_client("C1", {symbol: Quantity(0)}, 1000)
    client2 = market.new_client("C2", {symbol: Quantity(10)}, 1000)

    # Trade 1: C1 buys 2 @ 100 from C2
    for _ in range(2):
        client2.send_order(symbol, Side.SELL, Price(100))
        client1.send_order(symbol, Side.BUY, Price(100))

    # Trade 2: C1 buys 3 @ 110 from C2
    for _ in range(3):
        client2.send_order(symbol, Side.SELL, Price(110))
        client1.send_order(symbol, Side.BUY, Price(110))

    # Trade 3: C1 sells 1 @ 120 to C2
    client1.send_order(symbol, Side.SELL, Price(120))
    client2.send_order(symbol, Side.BUY, Price(120))

    # Expected values for Client 1
    # Bought: 2*100 + 3*110 = 200 + 330 = 530. Qty = 5.
    # Sold: 1*120 = 120. Qty = 1.
    # Net Cash Change = 120 - 530 = -410.
    # Final Cash = 1000 - 410 = 590.

    s1 = summarizer.summary[symbol][client1]
    assert s1.buy_quantity == 5
    assert s1.buy_consideration == 530
    assert s1.average_buy_price() == 106.0

    assert s1.sell_quantity == 1
    assert s1.sell_consideration == 120
    assert s1.average_sell_price() == 120.0

    assert client1.cash == 590

    # Verify formula
    cash_change = (
        s1.average_sell_price() * s1.sell_quantity
        - s1.average_buy_price() * s1.buy_quantity
    )
    # 120 * 1 - 106 * 5 = 120 - 530 = -410
    # Client cash change = 590 - 1000 = -410
    assert (client1.cash - 1000) == cash_change

    # Expected values for Client 2
    # Sold: 5 units. Consideration = 530. Avg Sell = 106.
    # Bought: 1 unit. Consideration = 120. Avg Buy = 120.
    # Net Cash Change = 530 - 120 = 410.
    # Final Cash = 1000 + 410 = 1410.

    s2 = summarizer.summary[symbol][client2]
    assert s2.sell_quantity == 5
    assert s2.sell_consideration == 530
    assert s2.average_sell_price() == 106.0

    assert s2.buy_quantity == 1
    assert s2.buy_consideration == 120
    assert s2.average_buy_price() == 120.0

    assert client2.cash == 1410

    # Verify formula
    cash_change_2 = (
        s2.average_sell_price() * s2.sell_quantity
        - s2.average_buy_price() * s2.buy_quantity
    )
    assert (client2.cash - 1000) == cash_change_2


def test_active_orders_consistency() -> None:
    market = Market([symbol])
    client = market.new_client("C1", {symbol: Quantity(10)})

    # Place 3 sell orders
    for _ in range(3):
        client.send_order(symbol, Side.SELL, Price(100))

    # Verify client tracks 3 active orders
    assert len(client.order_managers[symbol].asks.orders) == 3
    assert client.order_managers[symbol].asks.pending_orders == 0

    # Verify market tracks 3 orders
    book = market.books[symbol]
    queue = book.asks.limits[Price(100)]
    assert queue.total_quantity == 3

    # Verify orders match
    # We can iterate queue to get orders
    market_orders = set[Order]()
    current = queue.head.next
    while current != queue.tail:
        assert isinstance(current, ActiveOrderQueued)
        market_orders.add(current.order)
        current = current.next

    client_orders = {o.order for o in client.active_orders(symbol)}
    assert market_orders == client_orders

    # Cancel one order via client
    order_to_cancel = next(iter(client.active_orders(symbol)))
    client.cancel_order(order_to_cancel.order)

    assert len(list(client.active_orders(symbol))) == 2
    assert queue.total_quantity == 2

    # Verify remaining sells updated
    assert client.order_managers[symbol].asks.remaining_quantity == 2


def test_can_only_cancel_self_order() -> None:
    market = Market([symbol])
    client1 = market.new_client("C1", {symbol: Quantity(10)})
    client2 = market.new_client("C2", {symbol: Quantity(10)})

    order1a = client1.send_order(symbol, Side.SELL, Price(100))
    order1b = client1.send_order(symbol, Side.SELL, Price(100))
    order2 = client2.send_order(symbol, Side.SELL, Price(100))

    # Verify clients tracks all 3 active orders
    assert len(client1.order_managers[symbol].asks.orders) == 2
    assert len(client2.order_managers[symbol].asks.orders) == 1

    # Verify market tracks 3 orders
    book = market.books[symbol]
    queue = book.asks.limits[Price(100)]
    assert queue.total_quantity == 3

    # Client 1 cancels own orders
    client1.cancel_order(order1a)
    assert queue.total_quantity == 2

    # Client 1 cancels own orders again to check idempotency
    client1.cancel_order(order1a)
    assert queue.total_quantity == 2

    # Client 2 tries to cancel Client 1's cancelled orders
    with pytest.raises(OrderOperationNotAllowedError):
        client2.cancel_order(order1a)
    assert queue.total_quantity == 2

    # Client 2 tries to cancel Client 1's active orders
    with pytest.raises(OrderOperationNotAllowedError):
        client2.cancel_order(order1b)
    assert queue.total_quantity == 2

    # Client 2 cancels own order
    client2.cancel_order(order2)
    assert queue.total_quantity == 1

    # Client 1 cancels own order
    client1.cancel_order(order1b)
    assert queue.total_quantity == 0


def test_self_match_prevention() -> None:
    market = Market([symbol])
    client = market.new_client("C1", {symbol: Quantity(10)})

    # Place Buy @ 100
    client.send_order(symbol, Side.BUY, Price(100))

    # Try to Sell @ 100 -> Should fail (Self match)
    with pytest.raises(
        PotentialSelfMatchError,
    ):
        client.send_order(symbol, Side.SELL, Price(100))

    # Try to Sell @ 99 -> Should fail (Self match)
    with pytest.raises(
        PotentialSelfMatchError,
    ):
        client.send_order(symbol, Side.SELL, Price(99))

    # Sell @ 101 -> OK
    client.send_order(symbol, Side.SELL, Price(101))

    # Now Client has Buy @ 100, Sell @ 101

    # Try to Buy @ 101 -> Should fail (matches own Sell @ 101)
    with pytest.raises(
        PotentialSelfMatchError,
    ):
        client.send_order(symbol, Side.BUY, Price(101))

    # Try to Buy @ 102 -> Should fail (matches own Sell @ 101)
    with pytest.raises(
        PotentialSelfMatchError,
    ):
        client.send_order(symbol, Side.BUY, Price(102))

    # Buy @ 100 -> OK (adds to existing Buy @ 100)
    client.send_order(symbol, Side.BUY, Price(100))

    # Verify active orders
    # 2 Buys @ 100, 1 Sell @ 101
    assert len(client.order_managers[symbol].bids.orders) == 2
    assert len(client.order_managers[symbol].asks.orders) == 1

    # Test Cancellation and Re-entry
    # Current state: 2 Buys @ 100, 1 Sell @ 101

    # Try to Sell @ 100 -> Fail (Self match with Buys @ 100)
    with pytest.raises(
        PotentialSelfMatchError,
    ):
        client.send_order(symbol, Side.SELL, Price(100))

    # Cancel all Buys @ 100
    # We need to iterate and cancel. Note: iterating while modifying is unsafe, so list() first.
    orders_to_cancel = list(client.active_orders(symbol))
    for order in orders_to_cancel:
        if order.order.side == Side.BUY and order.order.price == 100:
            client.cancel_order(order.order)

    # Verify no Buys @ 100
    assert client.order_managers[symbol].bids.top_price() is None

    # Now Sell @ 100 -> Should succeed
    client.send_order(symbol, Side.SELL, Price(100))

    # Verify active orders: 1 Sell @ 101, 1 Sell @ 100
    assert len(client.order_managers[symbol].asks.orders) == 2
    assert client.order_managers[symbol].asks.top_price() == 100


def test_short_sell_error_on_execution() -> None:
    market = Market([symbol])
    client = market.new_client("C1")
    # Position is 0 by default

    # Manually create order and insert to bypass validation
    order = Order(client, symbol, Side.SELL, Price(100))
    client.order_managers[symbol].insert(order)

    # Notify the order is indeed filled and results in short sell
    with pytest.raises(ShortSellError):
        client.on_new_order_update(
            OrderStatusUpdate(
                order,
                OrderStatus.FILLED,
                Quantity(0),
                Quantity(1),
                100,
            ),
        )


def test_ioc_client_full_fill() -> None:
    market = Market([symbol])
    client1 = market.new_client("C1", {symbol: Quantity(10)})
    client2 = market.new_client("C2", {symbol: Quantity(10)})

    # C1 places Sell @ 100
    client1.send_order(symbol, Side.SELL, Price(100))

    # C2 places IOC Buy @ 100
    client2.send_order(
        symbol,
        Side.BUY,
        Price(100),
        order_type=OrderType.IMMEDIATE_OR_CANCEL,
    )

    # Check positions
    # C1 sold 1
    assert client1.order_managers[symbol].quantity == 9
    # C2 bought 1
    assert client2.order_managers[symbol].quantity == 11

    # C2 should have NO active orders (IOC didn't rest)
    assert list(client2.active_orders(symbol)) == []
    # C1 should have NO active orders (filled)
    assert list(client1.active_orders(symbol)) == []


def test_ioc_client_partial_fill() -> None:
    market = Market([symbol])
    client1 = market.new_client("C1", {symbol: Quantity(10)})
    client2 = market.new_client("C2", {symbol: Quantity(10)})

    # C1 places Sell @ 100 (Qty 1)
    client1.send_order(symbol, Side.SELL, Price(100))

    # C2 places IOC Buy @ 100 (Qty 2)
    client2.send_order(
        symbol,
        Side.BUY,
        Price(100),
        # quantity=Quantity(2),  # noqa: ERA001
        order_type=OrderType.IMMEDIATE_OR_CANCEL,
    )

    # Check positions
    # C1 sold 1
    assert client1.order_managers[symbol].quantity == 9
    # C2 bought 1 (partial fill)
    assert client2.order_managers[symbol].quantity == 11

    # C2 should have NO active orders (IOC remainder cancelled)
    assert list(client2.active_orders(symbol)) == []
    # C1 should have NO active orders (filled)
    assert list(client1.active_orders(symbol)) == []


def test_ioc_client_no_fill() -> None:
    market = Market([symbol])
    client1 = market.new_client("C1", {symbol: Quantity(10)})
    client2 = market.new_client("C2", {symbol: Quantity(10)})

    # C1 places Sell @ 101
    client1.send_order(symbol, Side.SELL, Price(101))

    # C2 places IOC Buy @ 100
    client2.send_order(
        symbol,
        Side.BUY,
        Price(100),
        order_type=OrderType.IMMEDIATE_OR_CANCEL,
    )

    # Check positions (unchanged)
    assert client1.order_managers[symbol].quantity == 10
    assert client2.order_managers[symbol].quantity == 10

    # C2 should have NO active orders
    assert list(client2.active_orders(symbol)) == []
    # C1 should have 1 active order
    assert len(list(client1.active_orders(symbol))) == 1


def test_min_net_quantity_observed() -> None:
    market = Market([symbol])
    client1 = market.new_client("C1", {symbol: Quantity(10)})
    client2 = market.new_client("C2", {symbol: Quantity(10)})

    # C1 sells, C2 buys
    client1.send_order(symbol, Side.SELL, Price(101))
    client2.send_order(symbol, Side.BUY, Price(101))

    assert (
        market.trade_summary.summary[symbol][client1].min_net_quantity_change
        == -1
    )
    assert (
        market.trade_summary.summary[symbol][client2].min_net_quantity_change
        == 0
    )

    # C1 buys, C2 sells
    client1.send_order(symbol, Side.BUY, Price(101))
    client2.send_order(symbol, Side.SELL, Price(101))

    assert (
        market.trade_summary.summary[symbol][client1].min_net_quantity_change
        == -1
    )
    assert (
        market.trade_summary.summary[symbol][client2].min_net_quantity_change
        == 0
    )

    # C1 sells, C2 buys
    client1.send_order(symbol, Side.SELL, Price(101), Quantity(2))
    client2.send_order(symbol, Side.BUY, Price(101), Quantity(2))

    assert (
        market.trade_summary.summary[symbol][client1].min_net_quantity_change
        == -2
    )
    assert (
        market.trade_summary.summary[symbol][client2].min_net_quantity_change
        == 0
    )


def test_cancel_all_active_orders() -> None:
    market = Market([symbol])
    client = market.new_client("C1", {symbol: Quantity(10)})

    # Place several active orders on both sides
    for _ in range(3):
        client.send_order(symbol, Side.SELL, Price(100))
    for _ in range(2):
        client.send_order(symbol, Side.BUY, Price(90))

    # Ensure there are active orders
    assert len(list(client.active_orders(symbol))) == 5

    # Cancel all active orders via convenience method
    client.cancel_all_active_orders()

    # Client should have no active orders
    assert list(client.active_orders(symbol)) == []

    # Market book should be empty (no top of book)
    top = market.books[symbol].top()
    assert top.bid is None
    assert top.ask is None
