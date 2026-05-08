from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

from figgie_gym.market.common import (
    Order,
    OrderStatus,
    OrderStatusUpdate,
    Price,
    Quantity,
    Side,
    Trade,
)
from figgie_gym.market.exception import (
    InvalidOrderStatusError,
    QueueEmptyError,
)
from figgie_gym.market.order_fill import OrderInMarket
from figgie_gym.market.validate import validate_matchable


class QueuedItem:
    __slots__ = ("next", "prev")

    prev: QueuedItem
    next: QueuedItem

    @staticmethod
    def link(first: QueuedItem, second: QueuedItem) -> None:
        first.next = second
        second.prev = first

    def detach(self) -> None:
        QueuedItem.link(self.prev, self.next)
        del self.prev
        del self.next


class QueueHead(QueuedItem):
    __slots__ = ("status",)

    def __init__(self) -> None:
        self.prev = self
        self.status = OrderStatus.INVALID


class QueueTail(QueuedItem):
    __slots__ = ("status",)

    def __init__(self) -> None:
        self.next = self
        self.status = OrderStatus.INVALID


class ActiveOrderQueued(QueuedItem, OrderInMarket):
    """Order in a price limit queue that is not immedately matched or killed."""

    __slots__ = ("status",)

    def __init__(
        self,
        order: OrderInMarket,
        queue: PriceLimitQueue,
    ) -> None:
        if order.status is not OrderStatus.NEW:
            raise InvalidOrderStatusError(order.status)

        self.queue = queue
        self.order = order.order
        self.status = OrderStatus.ACTIVE
        self.remaining_quantity = order.remaining_quantity
        self.cumulative_quantity = order.cumulative_quantity
        self.cumulative_consideration = order.cumulative_consideration

    @classmethod
    def newly_enqueued(
        cls,
        order: OrderInMarket,
        prev: QueuedItem,
        next_: QueuedItem,
        queue: PriceLimitQueue,
    ) -> Self:
        new_self = cls(order, queue)
        new_self.remaining_quantity = order.remaining_quantity
        QueuedItem.link(prev, new_self)
        QueuedItem.link(new_self, next_)
        return new_self

    def cancel(self) -> OrderStatusUpdate:
        self.queue.adjust_quantity(Quantity(-self.remaining_quantity))
        self.zero_out(OrderStatus.CANCELED)
        self.detach()
        return self.make_update()


class QueueOwner(ABC):
    @abstractmethod
    def prune_price_level(self, price: Price) -> None: ...

    @abstractmethod
    def prune_active_order_for(
        self,
        order: Order,
    ) -> None: ...

    @abstractmethod
    def maybe_find_active_order_for(
        self,
        order: Order,
    ) -> ActiveOrderQueued | None: ...


class PriceLimitQueue:
    def __init__(self, queue_owner: QueueOwner, price: Price) -> None:
        self.queue_owner = queue_owner
        self.total_quantity = Quantity(0)
        self.head = QueueHead()
        self.tail = QueueTail()
        ActiveOrderQueued.link(self.head, self.tail)
        self.price = price

    def _create_queued_order(
        self,
        order: OrderInMarket,
        prev: QueuedItem,
        next_: QueuedItem,
    ) -> ActiveOrderQueued:
        self.adjust_quantity(Quantity(order.remaining_quantity))
        return ActiveOrderQueued.newly_enqueued(order, prev, next_, self)

    @property
    def first(self) -> ActiveOrderQueued:
        if not isinstance(self.head.next, ActiveOrderQueued):
            raise QueueEmptyError
        return self.head.next

    @property
    def last(self) -> ActiveOrderQueued:
        if not isinstance(self.tail.prev, ActiveOrderQueued):
            raise QueueEmptyError
        return self.tail.prev

    def adjust_quantity(self, delta: Quantity) -> None:
        self.total_quantity = Quantity(self.total_quantity + delta)
        if self.total_quantity == 0:
            self.queue_owner.prune_price_level(self.price)

    def match(
        self,
        new_order: OrderInMarket,
    ) -> tuple[list[Trade], list[OrderStatusUpdate]]:
        trades, order_status_updates = list[Trade](), list[OrderStatusUpdate]()
        while self.total_quantity > 0 and new_order.remaining_quantity > 0:
            new_trades, new_updates = self.match_first_order_in_queue(
                new_order,
                self.first,
            )
            trades.extend(new_trades)
            order_status_updates.extend(new_updates)
        return trades, order_status_updates

    def match_first_order_in_queue(
        self,
        new_order: OrderInMarket,
        resting_order: ActiveOrderQueued,
    ) -> tuple[list[Trade], list[OrderStatusUpdate]]:
        validate_matchable(new_order.order, resting_order.order)

        execution_price = resting_order.order.price
        executed_quantity = Quantity(
            min(new_order.remaining_quantity, resting_order.remaining_quantity),
        )

        new_order.execute_quantity(executed_quantity, execution_price)
        resting_order.execute_quantity(executed_quantity, execution_price)
        self.adjust_quantity(Quantity(-executed_quantity))

        if resting_order.status == OrderStatus.FILLED:
            resting_order.detach()
            self.queue_owner.prune_active_order_for(resting_order.order)

        order_status_updates = [
            resting_order.make_update(),
            new_order.make_update(),
        ]

        match new_order.order.side:
            case Side.BUY:
                buyer, seller = new_order.order, resting_order.order
            case Side.SELL:
                seller, buyer = new_order.order, resting_order.order

        trades = [
            Trade(
                buyer.client,
                seller.client,
                resting_order.order.symbol,
                execution_price,
                executed_quantity,
                buyer,
                seller,
                new_order.order.side,
            ),
        ]
        return trades, order_status_updates

    def enqueue(self, order: OrderInMarket) -> ActiveOrderQueued:
        return self._create_queued_order(order, self.tail.prev, self.tail)
