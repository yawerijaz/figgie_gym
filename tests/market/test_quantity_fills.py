import pytest

from figgie_gym.market.client import MarketClient
from figgie_gym.market.common import (
    OneSideQuote,
    Price,
    Quantity,
    Quote,
    Side,
    Symbol,
)
from figgie_gym.market.market import Market, TradeTickerTape

SYMBOL = Symbol("test")

# Helper to create a market pre-seeded with a predictable depth profile.
# The layout is intentionally symmetric and uses two resting participants (a and b)
# so tests can assert fills across multiple price levels and owners.


def fresh_market_setup(
    side: Side,
    best_price: Price,
) -> tuple[Market, TradeTickerTape, MarketClient, MarketClient, MarketClient]:
    """Set up a market for testing.

    Market depth as follow:
    best price: qty = [2(a), 2(b), 3(b)]  (total 7)
    2nd best price: qty = [4(a), 5(a), 6(b)] (total 15)
    3rd best price has no orders
    4th best price: qty = [4(b), 5(a), 6(a), 7(b)] (total 22)

    (a) and (b) indicate different owners of the resting orders.
    """
    market = Market([SYMBOL])
    ticker_tape = market.create_trade_ticker_tape()

    # Create clients
    resting_a = market.new_client("resting_a", {SYMBOL: Quantity(100)})
    resting_b = market.new_client("resting_b", {SYMBOL: Quantity(100)})
    hitter = market.new_client("hitter", {SYMBOL: Quantity(100)})

    # at price BEST, there are 3 resting sell orders, with quantities 2, 2, 3
    price = best_price
    resting_a.send_order(SYMBOL, side, price, quantity=Quantity(2))
    resting_b.send_order(SYMBOL, side, price, quantity=Quantity(2))
    resting_b.send_order(SYMBOL, side, price, quantity=Quantity(3))

    # at price BEST-1, there are 3 resting sell orders, with quantities 4, 5, 6
    price = Price(best_price - side)
    resting_a.send_order(SYMBOL, side, price, quantity=Quantity(4))
    resting_a.send_order(SYMBOL, side, price, quantity=Quantity(5))
    resting_b.send_order(SYMBOL, side, price, quantity=Quantity(6))

    # at price BEST-3, there are 4 resting sell orders, with quantities 4, 5, 6, 7
    price = Price(best_price - side * 3)
    resting_b.send_order(SYMBOL, side, price, quantity=Quantity(4))
    resting_a.send_order(SYMBOL, side, price, quantity=Quantity(5))
    resting_a.send_order(SYMBOL, side, price, quantity=Quantity(6))
    resting_b.send_order(SYMBOL, side, price, quantity=Quantity(7))

    return market, ticker_tape, resting_a, resting_b, hitter


both_sides_best_best_plus_one_scenario = [
    (Side.SELL, Side.BUY, Price(50), Price(50)),
    (Side.SELL, Side.BUY, Price(50), Price(51)),
    (Side.BUY, Side.SELL, Price(20), Price(20)),
    (Side.BUY, Side.SELL, Price(20), Price(19)),
]


# Parameter matrix used for tests that should behave the same for BUY/SELL
# when the hitting price is at or one tick beyond the best.
def make_quote(resting_side: Side, one_side_quote: OneSideQuote) -> Quote:
    match resting_side:
        case Side.BUY:
            return Quote(
                one_side_quote,
                None,
            )
        case Side.SELL:
            return Quote(
                None,
                one_side_quote,
            )


# Tests for interactions with the best price and the next price level (best-1)
# The group below exercises partial fills, full fills, and progressing across
# multiple resting orders at the best price level.
# Verify market spot/top updates correctly when a hitting order is partially
# filled and the remaining resting portion is cancelled. This ensures the book
# and clients' positions/cash are consistent after cancellation.
@pytest.mark.parametrize(
    ("resting_side", "hitting_side", "best_price", "hit_price"),
    both_sides_best_best_plus_one_scenario,
)
def test_fill_first_order_partially(
    resting_side: Side,
    hitting_side: Side,
    best_price: Price,
    hit_price: Price,
) -> None:
    market, ticker_tape, resting_a, _, hitter = fresh_market_setup(
        resting_side,
        best_price,
    )
    hitter.send_order(SYMBOL, hitting_side, hit_price, quantity=Quantity(1))

    assert len(ticker_tape.trades) == 1
    assert ticker_tape.trades[0].quantity == Quantity(1)

    assert hitter.order_managers[SYMBOL].quantity == 100 + hitting_side * 1
    assert resting_a.order_managers[SYMBOL].quantity == 100 + resting_side * 1

    assert (
        hitter.order_managers[SYMBOL].net_cash_movement
        == -hitting_side * 1 * best_price
    )
    assert (
        resting_a.order_managers[SYMBOL].net_cash_movement
        == -resting_side * 1 * best_price
    )
    assert market.books[SYMBOL].top() == make_quote(
        resting_side,
        OneSideQuote(best_price, Quantity(1 + 2 + 3)),
    )


@pytest.mark.parametrize(
    ("resting_side", "hitting_side", "best_price", "hit_price"),
    both_sides_best_best_plus_one_scenario,
)
def test_fill_first_order_fully(
    resting_side: Side,
    hitting_side: Side,
    best_price: Price,
    hit_price: Price,
) -> None:
    market, ticker_tape, resting_a, _, hitter = fresh_market_setup(
        resting_side,
        best_price,
    )
    hitter.send_order(SYMBOL, hitting_side, hit_price, quantity=Quantity(2))

    assert len(ticker_tape.trades) == 1
    assert ticker_tape.trades[0].quantity == Quantity(2)

    assert hitter.order_managers[SYMBOL].quantity == 100 + hitting_side * 2
    assert resting_a.order_managers[SYMBOL].quantity == 100 + resting_side * 2

    assert (
        hitter.order_managers[SYMBOL].net_cash_movement
        == -hitting_side * 2 * best_price
    )
    assert (
        resting_a.order_managers[SYMBOL].net_cash_movement
        == -resting_side * 2 * best_price
    )
    assert market.books[SYMBOL].top() == make_quote(
        resting_side,
        OneSideQuote(best_price, Quantity(2 + 3)),
    )


@pytest.mark.parametrize(
    ("resting_side", "hitting_side", "best_price", "hit_price"),
    both_sides_best_best_plus_one_scenario,
)
def test_fill_middle_order_partially(
    resting_side: Side,
    hitting_side: Side,
    best_price: Price,
    hit_price: Price,
) -> None:
    market, ticker_tape, resting_a, resting_b, hitter = fresh_market_setup(
        resting_side,
        best_price,
    )
    hitter.send_order(SYMBOL, hitting_side, hit_price, quantity=Quantity(3))

    assert len(ticker_tape.trades) == 2
    assert ticker_tape.trades[0].quantity == Quantity(2)
    assert ticker_tape.trades[1].quantity == Quantity(1)

    assert hitter.order_managers[SYMBOL].quantity == 100 + hitting_side * 3
    assert resting_a.order_managers[SYMBOL].quantity == 100 + resting_side * 2
    assert resting_b.order_managers[SYMBOL].quantity == 100 + resting_side * 1

    assert (
        hitter.order_managers[SYMBOL].net_cash_movement
        == -hitting_side * 3 * best_price
    )
    assert (
        resting_a.order_managers[SYMBOL].net_cash_movement
        == -resting_side * 2 * best_price
    )
    assert (
        resting_b.order_managers[SYMBOL].net_cash_movement
        == -resting_side * 1 * best_price
    )
    assert market.books[SYMBOL].top() == make_quote(
        resting_side,
        OneSideQuote(best_price, Quantity(1 + 3)),
    )


@pytest.mark.parametrize(
    ("resting_side", "hitting_side", "best_price", "hit_price"),
    both_sides_best_best_plus_one_scenario,
)
def test_fill_middle_order_fully(
    resting_side: Side,
    hitting_side: Side,
    best_price: Price,
    hit_price: Price,
) -> None:
    market, ticker_tape, resting_a, resting_b, hitter = fresh_market_setup(
        resting_side,
        best_price,
    )
    hitter.send_order(SYMBOL, hitting_side, hit_price, quantity=Quantity(4))

    assert len(ticker_tape.trades) == 2
    assert ticker_tape.trades[0].quantity == Quantity(2)
    assert ticker_tape.trades[1].quantity == Quantity(2)

    assert hitter.order_managers[SYMBOL].quantity == 100 + hitting_side * 4
    assert resting_a.order_managers[SYMBOL].quantity == 100 + resting_side * 2
    assert resting_b.order_managers[SYMBOL].quantity == 100 + resting_side * 2

    assert (
        hitter.order_managers[SYMBOL].net_cash_movement
        == -hitting_side * 4 * best_price
    )
    assert (
        resting_a.order_managers[SYMBOL].net_cash_movement
        == -resting_side * 2 * best_price
    )
    assert (
        resting_b.order_managers[SYMBOL].net_cash_movement
        == -resting_side * 2 * best_price
    )
    assert market.books[SYMBOL].top() == make_quote(
        resting_side,
        OneSideQuote(best_price, Quantity(3)),
    )


@pytest.mark.parametrize(
    ("resting_side", "hitting_side", "best_price", "hit_price"),
    both_sides_best_best_plus_one_scenario,
)
def test_fill_last_order_partially(
    resting_side: Side,
    hitting_side: Side,
    best_price: Price,
    hit_price: Price,
) -> None:
    market, ticker_tape, resting_a, resting_b, hitter = fresh_market_setup(
        resting_side,
        best_price,
    )
    hitter.send_order(SYMBOL, hitting_side, hit_price, quantity=Quantity(5))

    assert len(ticker_tape.trades) == 3
    assert ticker_tape.trades[0].quantity == Quantity(2)
    assert ticker_tape.trades[1].quantity == Quantity(2)
    assert ticker_tape.trades[2].quantity == Quantity(1)

    assert hitter.order_managers[SYMBOL].quantity == 100 + hitting_side * 5
    assert resting_a.order_managers[SYMBOL].quantity == 100 + resting_side * 2
    assert resting_b.order_managers[SYMBOL].quantity == 100 + resting_side * 3

    assert (
        hitter.order_managers[SYMBOL].net_cash_movement
        == -hitting_side * 5 * best_price
    )
    assert (
        resting_a.order_managers[SYMBOL].net_cash_movement
        == -resting_side * 2 * best_price
    )
    assert (
        resting_b.order_managers[SYMBOL].net_cash_movement
        == -resting_side * 3 * best_price
    )
    assert market.books[SYMBOL].top() == make_quote(
        resting_side,
        OneSideQuote(best_price, Quantity(2)),
    )


@pytest.mark.parametrize(
    ("resting_side", "hitting_side", "best_price", "hit_price"),
    both_sides_best_best_plus_one_scenario,
)
def test_fill_last_order_fully(
    resting_side: Side,
    hitting_side: Side,
    best_price: Price,
    hit_price: Price,
) -> None:
    market, ticker_tape, resting_a, resting_b, hitter = fresh_market_setup(
        resting_side,
        best_price,
    )
    hitter.send_order(SYMBOL, hitting_side, hit_price, quantity=Quantity(7))

    assert len(ticker_tape.trades) == 3
    assert ticker_tape.trades[0].quantity == Quantity(2)
    assert ticker_tape.trades[1].quantity == Quantity(2)
    assert ticker_tape.trades[2].quantity == Quantity(3)

    assert hitter.order_managers[SYMBOL].quantity == 100 + hitting_side * 7
    assert resting_a.order_managers[SYMBOL].quantity == 100 + resting_side * 2
    assert resting_b.order_managers[SYMBOL].quantity == 100 + resting_side * 5

    assert (
        hitter.order_managers[SYMBOL].net_cash_movement
        == -hitting_side * 7 * best_price
    )
    assert (
        resting_a.order_managers[SYMBOL].net_cash_movement
        == -resting_side * 2 * best_price
    )
    assert (
        resting_b.order_managers[SYMBOL].net_cash_movement
        == -resting_side * 5 * best_price
    )
    assert market.books[SYMBOL].top() == make_quote(
        resting_side,
        OneSideQuote(Price(best_price - resting_side), Quantity(15)),
    )


@pytest.mark.parametrize(
    ("resting_side", "hitting_side", "best_price", "hit_price"),
    [
        (Side.SELL, Side.BUY, Price(50), Price(50)),
        (Side.BUY, Side.SELL, Price(20), Price(20)),
    ],
)
def test_fill_last_order_fully_with_remainder(
    resting_side: Side,
    hitting_side: Side,
    best_price: Price,
    hit_price: Price,
) -> None:
    market, ticker_tape, resting_a, resting_b, hitter = fresh_market_setup(
        resting_side,
        best_price,
    )
    hitter.send_order(SYMBOL, hitting_side, hit_price, quantity=Quantity(10))

    assert len(ticker_tape.trades) == 3
    assert ticker_tape.trades[0].quantity == Quantity(2)
    assert ticker_tape.trades[1].quantity == Quantity(2)
    assert ticker_tape.trades[2].quantity == Quantity(3)

    assert hitter.order_managers[SYMBOL].quantity == 100 + hitting_side * 7
    assert resting_a.order_managers[SYMBOL].quantity == 100 + resting_side * 2
    assert resting_b.order_managers[SYMBOL].quantity == 100 + resting_side * 5

    assert (
        hitter.order_managers[SYMBOL].net_cash_movement
        == -hitting_side * 7 * best_price
    )
    assert (
        resting_a.order_managers[SYMBOL].net_cash_movement
        == -resting_side * 2 * best_price
    )
    assert (
        resting_b.order_managers[SYMBOL].net_cash_movement
        == -resting_side * 5 * best_price
    )
    match resting_side:
        case Side.BUY:
            assert market.books[SYMBOL].top() == Quote(
                OneSideQuote(Price(best_price - resting_side), Quantity(15)),
                OneSideQuote(hit_price, Quantity(3)),
            )
        case Side.SELL:
            assert market.books[SYMBOL].top() == Quote(
                OneSideQuote(hit_price, Quantity(3)),
                OneSideQuote(Price(best_price - resting_side), Quantity(15)),
            )


@pytest.mark.parametrize(
    ("resting_side", "hitting_side", "best_price", "hit_price"),
    [
        (Side.SELL, Side.BUY, Price(50), Price(51)),
        (Side.BUY, Side.SELL, Price(20), Price(19)),
    ],
)
def test_fill_entire_best_price_and_partial_next(
    resting_side: Side,
    hitting_side: Side,
    best_price: Price,
    hit_price: Price,
) -> None:
    market, ticker_tape, resting_a, resting_b, hitter = fresh_market_setup(
        resting_side,
        best_price,
    )
    hitter.send_order(SYMBOL, hitting_side, hit_price, quantity=Quantity(10))

    assert len(ticker_tape.trades) == 4
    assert ticker_tape.trades[0].quantity == Quantity(2)
    assert ticker_tape.trades[1].quantity == Quantity(2)
    assert ticker_tape.trades[2].quantity == Quantity(3)
    assert ticker_tape.trades[3].quantity == Quantity(3)

    assert hitter.order_managers[SYMBOL].quantity == 100 + hitting_side * 10
    assert resting_a.order_managers[SYMBOL].quantity == 100 + resting_side * 5
    assert resting_b.order_managers[SYMBOL].quantity == 100 + resting_side * 5

    assert hitter.order_managers[
        SYMBOL
    ].net_cash_movement == -hitting_side * 7 * best_price - hitting_side * 3 * (
        best_price - resting_side
    )
    assert resting_a.order_managers[
        SYMBOL
    ].net_cash_movement == -resting_side * 2 * best_price - resting_side * 3 * (
        best_price - resting_side
    )
    assert (
        resting_b.order_managers[SYMBOL].net_cash_movement
        == -resting_side * 5 * best_price
    )
    assert market.books[SYMBOL].top() == make_quote(
        resting_side,
        OneSideQuote(Price(best_price - resting_side), Quantity(12)),
    )


@pytest.mark.parametrize(
    ("resting_side", "hitting_side", "best_price", "hit_price"),
    [
        (Side.SELL, Side.BUY, Price(50), Price(51)),
        (Side.BUY, Side.SELL, Price(20), Price(19)),
    ],
)
def test_fill_entire_best_price_and_full_next(
    resting_side: Side,
    hitting_side: Side,
    best_price: Price,
    hit_price: Price,
) -> None:
    market, ticker_tape, resting_a, resting_b, hitter = fresh_market_setup(
        resting_side,
        best_price,
    )
    hitter.send_order(SYMBOL, hitting_side, hit_price, quantity=Quantity(22))

    assert len(ticker_tape.trades) == 6
    assert ticker_tape.trades[0].quantity == Quantity(2)
    assert ticker_tape.trades[1].quantity == Quantity(2)
    assert ticker_tape.trades[2].quantity == Quantity(3)
    assert ticker_tape.trades[3].quantity == Quantity(4)
    assert ticker_tape.trades[4].quantity == Quantity(5)
    assert ticker_tape.trades[5].quantity == Quantity(6)

    assert hitter.order_managers[SYMBOL].quantity == 100 + hitting_side * 22
    assert resting_a.order_managers[SYMBOL].quantity == 100 + resting_side * 11
    assert resting_b.order_managers[SYMBOL].quantity == 100 + resting_side * 11

    assert (
        hitter.order_managers[SYMBOL].net_cash_movement
        == -hitting_side * 7 * best_price
        - hitting_side * 15 * (best_price - resting_side)
    )
    assert resting_a.order_managers[
        SYMBOL
    ].net_cash_movement == -resting_side * 2 * best_price - resting_side * 9 * (
        best_price - resting_side
    )
    assert resting_b.order_managers[
        SYMBOL
    ].net_cash_movement == -resting_side * 5 * best_price - resting_side * 6 * (
        best_price - resting_side
    )
    assert market.books[SYMBOL].top() == make_quote(
        resting_side,
        OneSideQuote(Price(best_price - 3 * resting_side), Quantity(22)),
    )


@pytest.mark.parametrize(
    ("resting_side", "hitting_side", "best_price", "hit_price"),
    [
        (Side.SELL, Side.BUY, Price(50), Price(55)),
        (Side.BUY, Side.SELL, Price(20), Price(15)),
    ],
)
# This test simulates a very large sweeping order that walks the book and leaves
# a remainder resting at the sweep price; it then exercises a counter-order that
# reclaims the remainder and leaves residual resting quantity at the original
# best price. It verifies positions, active orders and the market top.
def test_big_sweeping_orders(
    resting_side: Side,
    hitting_side: Side,
    best_price: Price,
    hit_price: Price,
) -> None:
    market, ticker_tape, _, _, hitter = fresh_market_setup(
        resting_side,
        best_price,
    )
    hitter.send_order(SYMBOL, hitting_side, hit_price, quantity=Quantity(100))

    assert (
        market.trade_summary.last_prices[SYMBOL]
        == best_price - resting_side * 3
    )
    # Should have filled 44
    assert hitter.order_managers[SYMBOL].quantity == 100 + hitting_side * 44
    # Remaining 56 should be resting at hitting price
    active = list(hitter.active_orders(SYMBOL))
    assert len(active) == 1
    assert active[0].remaining_quantity == 56
    assert active[0].order.price == hit_price

    assert market.books[SYMBOL].top() == make_quote(
        hitting_side,
        OneSideQuote(hit_price, Quantity(56)),
    )

    # Then a big counter order for 100 at original best price
    counter_hitter = market.new_client(
        "counter_hitter",
        {SYMBOL: Quantity(100)},
    )
    counter_order = counter_hitter.send_order(
        SYMBOL,
        resting_side,
        best_price,
        quantity=Quantity(100),
    )

    assert market.trade_summary.last_prices[SYMBOL] == hit_price
    # Should match the 56 resting at hit price
    # Hitter position: 44 + 56 = 100
    assert hitter.order_managers[SYMBOL].quantity == 100 + hitting_side * 100
    # Counter hitter position: 100 +/- 54 = 0
    assert (
        counter_hitter.order_managers[SYMBOL].quantity
        == 100 + resting_side * 56
    )

    # Counter hitter should have 100 - 56 = 44 resting at original best price
    active_counter = list(counter_hitter.active_orders(SYMBOL))
    assert len(active_counter) == 1
    assert active_counter[0].remaining_quantity == 44
    assert active_counter[0].order.price == best_price

    assert market.books[SYMBOL].top() == make_quote(
        resting_side,
        OneSideQuote(best_price, Quantity(44)),
    )

    assert len(ticker_tape.trades) == 11
    counter_hitter.cancel_order(counter_order)

    # Hitter disposes inventory so we can check cash movements
    hitter.send_order(SYMBOL, resting_side, hit_price, quantity=Quantity(100))
    counter_hitter.send_order(
        SYMBOL,
        hitting_side,
        hit_price,
        quantity=Quantity(100),
    )

    assert (
        hitter.order_managers[SYMBOL].net_cash_movement
        == 7 * 5 + 15 * 4 + 22 * 2
    )
    assert hitter.order_managers[SYMBOL].quantity == 100


@pytest.mark.parametrize(
    ("resting_side", "hitting_side", "best_price", "hit_price"),
    [
        (Side.SELL, Side.BUY, Price(50), Price(50)),
        (Side.BUY, Side.SELL, Price(20), Price(20)),
    ],
)
def test_market_spot_after_cancellation_and_partial_fill(
    resting_side: Side,
    hitting_side: Side,
    best_price: Price,
    hit_price: Price,
) -> None:
    market, ticker_tape, resting_a, resting_b, hitter = fresh_market_setup(
        resting_side,
        best_price,
    )

    # Place a large hitting order that will consume the entire best price
    # level (total 7) and leave 3 remaining resting at the hit price.
    order = hitter.send_order(
        SYMBOL,
        hitting_side,
        hit_price,
        quantity=Quantity(10),
    )

    # Should have produced 3 trades matching the best price level: 2,2,3
    assert len(ticker_tape.trades) == 3
    assert ticker_tape.trades[0].quantity == Quantity(2)
    assert ticker_tape.trades[1].quantity == Quantity(2)
    assert ticker_tape.trades[2].quantity == Quantity(3)

    # Only 7 units filled; position updated accordingly
    assert hitter.order_managers[SYMBOL].quantity == 100 + hitting_side * 7
    assert resting_a.order_managers[SYMBOL].quantity == 100 + resting_side * 2
    assert resting_b.order_managers[SYMBOL].quantity == 100 + resting_side * 5

    # Cash movements reflect the 7 filled at best_price
    assert (
        hitter.order_managers[SYMBOL].net_cash_movement
        == -hitting_side * 7 * best_price
    )
    assert (
        resting_a.order_managers[SYMBOL].net_cash_movement
        == -resting_side * 2 * best_price
    )
    assert (
        resting_b.order_managers[SYMBOL].net_cash_movement
        == -resting_side * 5 * best_price
    )

    # The hitter should have 1 active resting order for the remaining 3 at hit_price
    active = list(hitter.active_orders(SYMBOL))
    assert len(active) == 1
    assert active[0].remaining_quantity == Quantity(3)
    assert active[0].order.price == hit_price

    # Now cancel the remaining portion and ensure it's removed from the book
    hitter.cancel_order(order)

    assert list(hitter.active_orders(SYMBOL)) == []

    # After cancellation, the market top should show the next resting price level
    # (the second-best price level from the initial setup) on the resting side
    assert market.books[SYMBOL].top() == make_quote(
        resting_side,
        OneSideQuote(Price(best_price - resting_side), Quantity(15)),
    )
