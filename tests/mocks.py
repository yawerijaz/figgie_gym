from figgie_gym.market.client import MarketClient
from figgie_gym.market.common import (
    MarketTime,
    Order,
    Price,
    Quote,
    Symbol,
    Trade,
)
from figgie_gym.market.order_queue import ActiveOrderQueued, QueueOwner


class MockClient(MarketClient):
    def __init__(self, client_id: str) -> None:
        self.client_id = client_id
        self.trades = list[Trade]()
        self.positions = {}

    def on_new_trades(self, time: MarketTime, trades: list[Trade]) -> None:  # noqa: ARG002
        self.trades.extend(trades)

    def on_new_spot(self, symbol: Symbol, quote: Quote) -> None:
        pass

    def __repr__(self) -> str:
        return f"MockClient({self.client_id})"


class MockQueueOwner(QueueOwner):
    def __init__(self) -> None:
        self.pruned_prices = list[Price]()
        self.pruned_orders = list[Order]()

    def prune_price_level(self, price: Price) -> None:
        self.pruned_prices.append(price)

    def maybe_find_active_order_for(
        self,
        order: Order,
    ) -> ActiveOrderQueued | None: ...

    def prune_active_order_for(self, order: Order) -> None:
        self.pruned_orders.append(order)
