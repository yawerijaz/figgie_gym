from figgie_gym.market.common import Price, Quantity, Side, Symbol
from figgie_gym.market.market import Market

SYMBOL = Symbol("test")


def test_empty_spot() -> None:
    m = Market([SYMBOL])
    spot = m.spotter.spots[SYMBOL]
    assert spot.bid is None
    assert spot.ask is None


def test_only_bid_shows_bid() -> None:
    m = Market([SYMBOL])
    client = m.new_client("buyer")
    client.send_order(SYMBOL, Side.BUY, Price(100), quantity=Quantity(5))

    spot = m.spotter.spots[SYMBOL]
    assert spot.bid is not None
    assert spot.ask is None
    assert spot.bid.price == Price(100)
    assert spot.bid.quantity == Quantity(5)


def test_only_ask_shows_ask() -> None:
    m = Market([SYMBOL])
    client = m.new_client("seller", initial_positions={SYMBOL: Quantity(7)})
    client.send_order(SYMBOL, Side.SELL, Price(200), quantity=Quantity(7))

    spot = m.spotter.spots[SYMBOL]
    assert spot.ask is not None
    assert spot.bid is None
    assert spot.ask.price == Price(200)
    assert spot.ask.quantity == Quantity(7)


def test_bid_and_ask_shown_together() -> None:
    m = Market([SYMBOL])
    b = m.new_client("b")
    a = m.new_client("a", initial_positions={SYMBOL: Quantity(3)})
    b.send_order(SYMBOL, Side.BUY, Price(90), quantity=Quantity(2))
    a.send_order(SYMBOL, Side.SELL, Price(110), quantity=Quantity(3))

    spot = m.spotter.spots[SYMBOL]
    assert spot.bid is not None
    assert spot.ask is not None
    assert spot.bid.price == Price(90)
    assert spot.ask.price == Price(110)


def test_trade_updates_spot_when_book_empties() -> None:
    m = Market([SYMBOL])
    seller = m.new_client("s", initial_positions={SYMBOL: Quantity(1)})
    buyer = m.new_client("b")

    # seller posts 1 @100
    seller.send_order(SYMBOL, Side.SELL, Price(100), quantity=Quantity(1))
    # buyer crosses and fills it
    buyer.send_order(SYMBOL, Side.BUY, Price(100), quantity=Quantity(1))

    spot = m.spotter.spots[SYMBOL]
    # after the trade both sides should be empty
    assert spot.bid is None
    assert spot.ask is None


def test_cancel_updates_spot() -> None:
    m = Market([SYMBOL])
    client = m.new_client("c")
    order = client.send_order(SYMBOL, Side.BUY, Price(50), quantity=Quantity(4))

    spot = m.spotter.spots[SYMBOL]
    assert spot.bid is not None

    client.cancel_order(order)

    spot = m.spotter.spots[SYMBOL]
    assert spot.bid is None
    assert spot.ask is None


def test_partial_fill_updates_spot_quantity() -> None:
    m = Market([SYMBOL])
    seller = m.new_client("s", initial_positions={SYMBOL: Quantity(5)})
    buyer = m.new_client("b")

    # seller posts 5 @100
    seller.send_order(SYMBOL, Side.SELL, Price(100), quantity=Quantity(5))
    # buyer posts 2 @100 and should partially fill seller
    buyer.send_order(SYMBOL, Side.BUY, Price(100), quantity=Quantity(2))

    spot = m.spotter.spots[SYMBOL]
    assert spot.ask is not None
    # remaining should be 3
    assert spot.ask.quantity == Quantity(3)
