from __future__ import annotations

from abc import abstractmethod

from sortedcontainers import SortedDict

from figgie_gym.market.common import (
    OneSideQuote,
    Order,
    OrderStatus,
    OrderStatusUpdate,
    OrderType,
    Price,
    Quantity,
    Quote,
    Side,
    Trade,
)
from figgie_gym.market.exception import ImpossibleMarketSituationError
from figgie_gym.market.order_fill import OrderInMarket
from figgie_gym.market.order_queue import (
    ActiveOrderQueued,
    PriceLimitQueue,
    QueueOwner,
)


class OrderBookSide(QueueOwner):
    def __init__(self) -> None:
        self.limits = SortedDict[Price, PriceLimitQueue]()
        self.active_orders = dict[Order, ActiveOrderQueued]()

    def __bool__(self) -> bool:
        return bool(self.limits)

    @abstractmethod
    def is_matchable(self, order: Order) -> bool: ...

    @abstractmethod
    def best(self) -> tuple[Price, PriceLimitQueue]: ...

    def best_quote(self) -> OneSideQuote | None:
        if not self.limits:
            return None
        best_price, _ = self.best()
        return OneSideQuote(best_price, self.limits[best_price].total_quantity)

    def match(
        self,
        order: OrderInMarket,
    ) -> tuple[list[Trade], list[OrderStatusUpdate]]:
        trades, order_updates = list[Trade](), list[OrderStatusUpdate]()
        while order.remaining_quantity > 0 and self.is_matchable(order.order):
            _, queue = self.best()
            level_trades, level_order_updates = queue.match(order)
            trades.extend(level_trades)
            order_updates.extend(level_order_updates)
        return trades, order_updates

    def insert(self, order: OrderInMarket) -> OrderStatusUpdate:
        active_order = self.limits.setdefault(
            order.order.price,
            PriceLimitQueue(self, order.order.price),
        ).enqueue(order)
        self.active_orders[order.order] = active_order
        return active_order.make_update()

    def prune_price_level(self, price: Price) -> None:
        self.limits.pop(price, None)

    def maybe_find_active_order_for(
        self,
        order: Order,
    ) -> ActiveOrderQueued | None:
        return self.active_orders.get(order)

    def prune_active_order_for(self, order: Order) -> None:
        del self.active_orders[order]

    def depths(self) -> list[tuple[Price, Quantity]]:
        return [(p, q.total_quantity) for p, q in self.limits.items()]


class OrderBookBids(OrderBookSide):
    def best(self) -> tuple[Price, PriceLimitQueue]:
        return self.limits.items()[-1]

    def is_matchable(self, order: Order) -> bool:
        if not self:
            return False
        best_price, _ = self.best()
        return order.side is Side.SELL and order.price <= best_price


class OrderBookAsks(OrderBookSide):
    def best(self) -> tuple[Price, PriceLimitQueue]:
        return self.limits.items()[0]

    def is_matchable(self, order: Order) -> bool:
        if not self:
            return False
        best_price, _ = self.best()
        return order.side is Side.BUY and order.price >= best_price


class OrderBook:
    def __init__(self) -> None:
        self.bids = OrderBookBids()
        self.asks = OrderBookAsks()

    def process_order(
        self,
        order: Order,
    ) -> tuple[list[Trade], list[OrderStatusUpdate]]:
        """May have more than 1 order_updates for an order. Sequence not guaranteed."""
        match order.side:
            case Side.BUY:
                same_side, other_side = self.bids, self.asks
            case Side.SELL:
                same_side, other_side = self.asks, self.bids

        new_order = OrderInMarket.new(order)
        trades, order_updates = other_side.match(new_order)
        if new_order.status is OrderStatus.FILLED:
            return trades, order_updates
        unmatched_portion_update = (
            new_order.zero_out(OrderStatus.DEAD).make_update()
            if order.order_type == OrderType.IMMEDIATE_OR_CANCEL
            else same_side.insert(new_order)
        )
        return trades, [*order_updates, unmatched_portion_update]

    def cancel_order(self, order: Order) -> OrderStatusUpdate | None:
        match order.side:
            case Side.BUY:
                side = self.bids
            case Side.SELL:
                side = self.asks
        maybe_active_order = side.maybe_find_active_order_for(
            order,
        )
        if not maybe_active_order:
            return None
        side.prune_active_order_for(order)
        return maybe_active_order.cancel()

    def top(self) -> Quote:
        best_bid = self.bids.best_quote()
        best_ask = self.asks.best_quote()
        if (
            best_bid is not None
            and best_ask is not None
            and best_bid.price >= best_ask.price
        ):
            raise ImpossibleMarketSituationError(best_bid, best_ask)
        return Quote(best_bid, best_ask)

    def ladder(
        self,
    ) -> SortedDict[Price, tuple[Quantity | None, Quantity | None]]:
        bid_depths = self.bids.depths()
        ask_depths: list[tuple[Price, Quantity]] = self.asks.depths()
        ladder = SortedDict[Price, tuple[Quantity | None, Quantity | None]]()
        for p, qb in bid_depths:
            _, qa = ladder.get(p, (None, None))
            ladder[p] = (qb, qa)
        for p, qa in ask_depths:
            qb, _ = ladder.get(p, (None, None))
            ladder[p] = (qb, qa)

        return ladder
