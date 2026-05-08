from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import count
from typing import TYPE_CHECKING, Any, ClassVar

from sortedcontainers import SortedDict

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
    OrderOperationNotAllowedError,
    PotentialSelfMatchError,
    PotentialShortSellError,
    ShortSellError,
)
from figgie_gym.market.order_fill import OrderTransmitted

if TYPE_CHECKING:
    from collections.abc import Generator, Hashable, Iterator

    from figgie_gym.market.market import Market


class OrderManagerSide(ABC):
    def __init__(self, position: OrderManager) -> None:
        self.position = position
        self.orders = SortedDict[tuple[Price, int], OrderTransmitted]()

        self.pending_orders = 0
        self.pending_quantity = 0

        self.cumulative_quantity = 0
        self.remaining_quantity = 0
        self.cumulative_consideration = 0

    @abstractmethod
    def top_price(self) -> Price | None: ...

    def insert(self, order: Order) -> None:
        self.pending_quantity += order.quantity
        self.pending_orders += 1
        self.orders[(order.price, order.order_id)] = (
            OrderTransmitted.newly_transmitted(order)
        )

    def reconcile(self, order_update: OrderStatusUpdate) -> None:
        order_transmitted = self.orders.get(
            (order_update.order.price, order_update.order.order_id),
        )
        if not order_transmitted:
            return

        self.pending_quantity -= order_transmitted.pending_quantity
        self.pending_orders -= 1

        self.remaining_quantity += (
            -order_transmitted.remaining_quantity
            + order_update.remaining_quantity
        )
        self.cumulative_quantity += (
            -order_transmitted.cumulative_quantity
            + order_update.cumulative_quantity
        )
        self.cumulative_consideration += (
            -order_transmitted.cumulative_consideration
            + order_update.cumulative_consideration
        )
        self.position.refresh()

        order_transmitted.pending_quantity = Quantity(0)
        order_transmitted.remaining_quantity = order_update.remaining_quantity
        order_transmitted.cumulative_quantity = order_update.cumulative_quantity
        order_transmitted.cumulative_consideration = (
            order_update.cumulative_consideration
        )

        if order_update.status not in (OrderStatus.ACTIVE, OrderStatus.NEW):
            self.orders.pop(
                (order_update.order.price, order_update.order.order_id),
            )

    def iterate_orders(self) -> Generator[OrderTransmitted, Any]:
        yield from self.orders.values()


class OrderManagerBid(OrderManagerSide):
    def top_price(self) -> Price | None:
        if not self.orders:
            return None
        return self.orders.peekitem()[0][0]

    def could_potentially_self_match(
        self,
        order: Order,
    ) -> PotentialSelfMatchError | None:
        top = self.top_price()
        if order.side is Side.SELL and top is not None and order.price <= top:
            return PotentialSelfMatchError(order.price, top)
        return None


class OrderManagerAsk(OrderManagerSide):
    def top_price(self) -> Price | None:
        if not self.orders:
            return None
        return self.orders.peekitem(0)[0][0]

    def could_potentially_short_sell(
        self,
        order: Order,
    ) -> PotentialShortSellError | None:
        if (
            order.side is Side.SELL
            and (
                q := Quantity(
                    self.position.quantity
                    - self.pending_quantity
                    - self.remaining_quantity
                    - order.quantity,
                )
            )
            < 0
        ):
            return PotentialShortSellError(q)
        return None

    def could_potentially_self_match(
        self,
        order: Order,
    ) -> PotentialSelfMatchError | None:
        top = self.top_price()
        if order.side is Side.BUY and top is not None and order.price >= top:
            return PotentialSelfMatchError(order.price, top)
        return None


class Position:
    """Holds position-related state and encapsulates refresh logic."""

    def __init__(
        self,
        initial_quantity: Quantity,
    ) -> None:
        self.initial_quantity = initial_quantity
        self.quantity = initial_quantity
        self.net_cash_movement = 0

    def refresh_from_sides(
        self,
        bids: OrderManagerBid,
        asks: OrderManagerAsk,
    ) -> None:
        self.net_cash_movement = (
            asks.cumulative_consideration - bids.cumulative_consideration
        )
        self.quantity = Quantity(
            self.initial_quantity
            + bids.cumulative_quantity
            - asks.cumulative_quantity,
        )
        if self.quantity < 0:
            raise ShortSellError(self.quantity)


class OrderManager:
    def __init__(self, position: Position) -> None:
        self.bids = OrderManagerBid(self)
        self.asks = OrderManagerAsk(self)
        self._position = position

    @property
    def quantity(self) -> Quantity:
        return self._position.quantity

    @property
    def net_cash_movement(self) -> int:
        return self._position.net_cash_movement

    def side(self, side: Side) -> OrderManagerSide:
        match side:
            case Side.BUY:
                return self.bids
            case Side.SELL:
                return self.asks

    def insert(self, order: Order) -> None:
        self.side(order.side).insert(order)

    def refresh(self) -> None:
        self._position.refresh_from_sides(self.bids, self.asks)

    def reconcile(self, order_update: OrderStatusUpdate) -> None:
        self.side(order_update.order.side).reconcile(order_update)

    def iterate_orders(self) -> Generator[OrderTransmitted, Any]:
        yield from self.bids.iterate_orders()
        yield from self.asks.iterate_orders()

    @abstractmethod
    def could_potentially_self_match(
        self,
        order: Order,
    ) -> PotentialSelfMatchError | None: ...

    def validate_order(self, order: Order) -> None:
        if ex := self.asks.could_potentially_short_sell(order):
            raise ex
        if ex := self.bids.could_potentially_self_match(order):
            raise ex
        if ex := self.asks.could_potentially_self_match(order):
            raise ex


class MarketClient(Client):
    _id_counter: ClassVar[Iterator[int]] = count()

    def __init__(
        self,
        market: Market,
        client_id: Hashable,
        initial_positions: dict[Symbol, Quantity] | None = None,
        initial_cash: int = 0,
    ) -> None:
        self.client_id = client_id or next(self._id_counter)
        self.market = market
        self.initial_cash = initial_cash
        self.initial_positions = initial_positions or {}
        self.order_managers = {
            sym: OrderManager(
                Position(self.initial_positions.get(sym, Quantity(0))),
            )
            for sym in market.books
        }

    one = Quantity(1)

    def send_order(
        self,
        symbol: Symbol,
        side: Side,
        price: Price,
        quantity: Quantity = one,
        *,
        order_type: OrderType = OrderType.LIMIT,
    ) -> Order:
        order = Order(
            self,
            symbol,
            side,
            price,
            quantity=Quantity(quantity),
            order_type=order_type,
        )

        self.order_managers[symbol].validate_order(order)
        self.order_managers[symbol].insert(order)
        self.market.process_order(order)
        return order

    def cancel_order(self, order: Order) -> None:
        if order.client is not self:
            reason = "Cannot cancel order not owned by this client."
            raise OrderOperationNotAllowedError(reason)
        self.market.cancel_order(order)

    def iterate_orders(self) -> Generator[OrderTransmitted, Any]:
        for manager in self.order_managers.values():
            yield from manager.iterate_orders()

    def cancel_all_active_orders(self) -> None:
        # iterate over a static list to avoid modifying the underlying
        # collection while iterating which can skip items
        for order in list(self.iterate_orders()):
            self.cancel_order(order.order)

    @property
    def cash(self) -> int:
        return self.initial_cash + sum(
            p.net_cash_movement for p in self.order_managers.values()
        )

    def active_orders(self, symbol: Symbol) -> Generator[OrderTransmitted, Any]:
        yield from self.order_managers[symbol].iterate_orders()

    def on_new_order_update(self, order_update: OrderStatusUpdate) -> None:
        self.order_managers[order_update.order.symbol].reconcile(order_update)
