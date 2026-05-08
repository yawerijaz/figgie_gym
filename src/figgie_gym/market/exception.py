from dataclasses import dataclass

from figgie_gym.market.common import OneSideQuote, Order, Price, Quantity


class FiggieGymError(Exception): ...


@dataclass
class ShortSellError(FiggieGymError):
    position: Quantity


@dataclass
class PotentialShortSellError(ShortSellError):
    position: Quantity


@dataclass
class UnmatchableTradeError(FiggieGymError):
    hitting_order: Order
    standing_order: Order


@dataclass
class PotentialSelfMatchError(FiggieGymError):
    hitting_order_price: Price
    standing_order_price: Price


@dataclass
class ImpossibleMarketSituationError(FiggieGymError):
    best_bid: OneSideQuote
    best_ask: OneSideQuote


class ImpossibleOrderSituationError(FiggieGymError): ...


@dataclass
class InvalidOrderStatusError(FiggieGymError):
    status: str


class OrderOperationNotAllowedError(FiggieGymError): ...


class QueueEmptyError(FiggieGymError): ...
