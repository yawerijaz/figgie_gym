from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from figgie_gym.market.common import (
    Order,
    OrderStatus,
    OrderStatusUpdate,
    Price,
    Quantity,
)
from figgie_gym.market.exception import (
    ImpossibleOrderSituationError,
    InvalidOrderStatusError,
)


class OrderFillProgress:
    """Container for tracking an Order's fill progress."""

    order: Order
    status: OrderStatus
    remaining_quantity: Quantity
    cumulative_quantity: Quantity
    cumulative_consideration: int

    def make_update(self) -> OrderStatusUpdate:
        return OrderStatusUpdate(
            order=self.order,
            status=self.status,
            remaining_quantity=self.remaining_quantity,
            cumulative_quantity=self.cumulative_quantity,
            cumulative_consideration=self.cumulative_consideration,
        )


@dataclass
class OrderInMarket(OrderFillProgress):
    """Order's fill progress on market."""

    order: Order
    status: OrderStatus
    remaining_quantity: Quantity
    cumulative_quantity: Quantity
    cumulative_consideration: int

    @classmethod
    def new(cls, order: Order) -> Self:
        """Create a new OrderInMarket as it arrives at the market. It is to be matched on existing orders."""
        return cls(
            order=order,
            status=OrderStatus.NEW,
            remaining_quantity=order.quantity,
            cumulative_quantity=Quantity(0),
            cumulative_consideration=0,
        )

    def execute_quantity(self, qty: Quantity, price: Price) -> None:
        if self.status not in (OrderStatus.ACTIVE, OrderStatus.NEW):
            raise InvalidOrderStatusError(self.status)
        self.remaining_quantity = Quantity(self.remaining_quantity - qty)
        self.cumulative_quantity = Quantity(self.cumulative_quantity + qty)
        self.cumulative_consideration += qty * price

        if self.remaining_quantity < 0:
            raise ImpossibleOrderSituationError(self)

        if self.remaining_quantity == 0:
            self.status = OrderStatus.FILLED

    def zero_out(self, new_status: OrderStatus) -> Self:
        # Technically new orders can't be zeroed out until they are active,
        # but this is mainly to facilitate easier testing.
        if self.status not in (OrderStatus.ACTIVE, OrderStatus.NEW):
            raise InvalidOrderStatusError(self.status)
        self.remaining_quantity = Quantity(0)
        self.status = new_status
        return self


@dataclass
class OrderTransmitted(OrderFillProgress):
    """Order's fill progress from client's perspective. Needs to be reconciled with updates from market."""

    order: Order
    status: OrderStatus
    pending_quantity: Quantity
    remaining_quantity: Quantity
    cumulative_quantity: Quantity
    cumulative_consideration: int

    @classmethod
    def newly_transmitted(cls, order: Order) -> OrderTransmitted:
        return OrderTransmitted(
            order=order,
            status=OrderStatus.PENDING,
            pending_quantity=order.quantity,
            remaining_quantity=Quantity(0),
            cumulative_quantity=Quantity(0),
            cumulative_consideration=0,
        )
