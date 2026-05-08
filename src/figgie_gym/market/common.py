from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from itertools import count
from typing import TYPE_CHECKING, ClassVar, Literal, NewType, Self

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterator

OrderId = NewType("OrderId", int)
TradeId = NewType("TradeId", int)
ExecutionId = NewType("ExecutionId", int)

Symbol = NewType("Symbol", str)
Price = NewType("Price", int)
Quantity = NewType("Quantity", int)
MarketTime = NewType("MarketTime", int)


class Client:
    client_id: Hashable

    def on_new_trades(self, time: MarketTime, trades: list[Trade]) -> None: ...
    def on_new_order(
        self,
        time: MarketTime,
        order: Order,
        event: Literal["insert", "cancel"],
    ) -> None: ...
    def on_new_spot(self, symbol: Symbol, quote: Quote) -> None: ...
    def on_new_order_update(
        self,
        order_update: OrderStatusUpdate,
    ) -> None: ...

    def __hash__(self) -> int:
        return hash(self.client_id)

    def __repr__(self) -> str:
        return f"Client({self.client_id})"


@dataclass
class OneSideQuote:
    price: Price
    quantity: Quantity


@dataclass
class Quote:
    bid: OneSideQuote | None
    ask: OneSideQuote | None


class Side(IntEnum):
    BUY = 1
    SELL = -1


class OrderType(StrEnum):
    LIMIT = "limit"
    IMMEDIATE_OR_CANCEL = "immediate_or_cancel"


class OrderStatus(StrEnum):
    PENDING = "pending"
    NEW = "new"
    ACTIVE = "active"
    CANCELED = "canceled"
    FILLED = "filled"
    DEAD = "dead"
    INVALID = "invalid"


@dataclass
class Order:
    client: Client
    symbol: Symbol
    side: Side
    price: Price
    quantity: Quantity = field(default=Quantity(1))
    _id_counter: ClassVar[Iterator[int]] = count()
    order_id: int = field(default_factory=_id_counter.__next__, init=False)

    order_type: OrderType = field(default=OrderType.LIMIT, kw_only=True)

    def __hash__(self) -> int:
        return hash(
            (
                self.client,
                self.symbol,
                self.side,
                self.price,
                self.quantity,
                self.order_id,
            ),
        )


@dataclass(frozen=True)
class OrderStatusUpdate:
    """An immutable snapshot of an Order's fill progress."""

    order: Order
    status: OrderStatus
    remaining_quantity: Quantity
    cumulative_quantity: Quantity
    cumulative_consideration: int


@dataclass
class Trade:
    buyer: Client
    seller: Client
    symbol: Symbol
    price: Price
    quantity: Quantity
    buy_order: Order
    sell_order: Order
    aggressive_side: Side
    _id_counter: ClassVar[Iterator[int]] = count()
    trade_id: int = field(default_factory=_id_counter.__next__, init=False)


@dataclass
class TradeSummary:
    buy_quantity: Quantity
    buy_consideration: int
    sell_quantity: Quantity
    sell_consideration: int
    min_net_quantity_change: Quantity

    def average_buy_price(self) -> float:
        return (
            self.buy_consideration / self.buy_quantity
            if self.buy_quantity > 0
            else float("nan")
        )

    def average_sell_price(self) -> float:
        return (
            self.sell_consideration / self.sell_quantity
            if self.sell_quantity > 0
            else float("nan")
        )

    def net_quantity_change(self) -> int:
        return self.buy_quantity - self.sell_quantity

    @classmethod
    def new(cls) -> Self:
        return cls(Quantity(0), 0, Quantity(0), 0, Quantity(0))

    def __add__(self, trade: Trade) -> TradeSummary:
        return TradeSummary(
            Quantity(self.buy_quantity + trade.quantity),
            self.buy_consideration + trade.quantity * trade.price,
            Quantity(self.sell_quantity),
            self.sell_consideration,
            self.min_net_quantity_change,
        )

    def __sub__(self, trade: Trade) -> TradeSummary:
        return TradeSummary(
            Quantity(self.buy_quantity),
            self.buy_consideration,
            Quantity(self.sell_quantity + trade.quantity),
            self.sell_consideration + trade.quantity * trade.price,
            min(
                self.min_net_quantity_change,
                Quantity(
                    self.buy_quantity - self.sell_quantity - trade.quantity,
                ),
            ),
        )
