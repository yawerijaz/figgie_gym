from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from figgie_gym.agent.common import AGENT_NAME_CODES
from figgie_gym.market.common import (
    Price,
    Quantity,
    Quote,
    Symbol,
    TradeSummary,
)

if TYPE_CHECKING:
    import numpy as np

type AgentID = int
type ActionType = dict[Suit, ActionOnSuit]
type BeliefType = dict[Suit, float]
type RewardType = float
type InfoType = dict[str, str]
type ExtraType = Any

type Suit = Symbol
type AgentAction = dict[Suit, ActionOnSuit]
type Cash = float
type Time = int
type RemainingTime = int


@dataclass
class ObsType:
    time: Time
    remaining_time: RemainingTime
    remaining_time_fraction: float
    cash: Cash
    per_suit: dict[Suit, ObsOnSuit]


@dataclass
class ObsOnSuit:
    market_quote: Quote
    last_price: Price | None
    volume: Quantity
    self_position: Quantity
    known_count: Quantity
    self_trade_summary: TradeSummary
    other_trade_summaries: list[TradeSummary]


@dataclass
class ActionOnSuit:
    quote_bid: float
    quote_ask: float
    snipe_bid: float
    snipe_ask: float


SUITS = [Symbol(s) for s in ["Spade", "Club", "Heart", "Diamond"]]
SUIT_TO_CODE = {k: i for i, k in enumerate(SUITS)}

SUIT_PARTNER_MAP = {
    Symbol("Spade"): Symbol("Club"),
    Symbol("Club"): Symbol("Spade"),
    Symbol("Heart"): Symbol("Diamond"),
    Symbol("Diamond"): Symbol("Heart"),
}


class Agent(ABC):
    @abstractmethod
    def act(
        self,
        random_number_generator: np.random.Generator,
        observation: ObsType,
    ) -> tuple[BeliefType, ActionType, ExtraType]: ...

    @abstractmethod
    def act_batch(
        self,
        random_number_generator: np.random.Generator,
        observations: list[ObsType],
    ) -> list[tuple[BeliefType, ActionType, ExtraType]]:
        return [self.act(random_number_generator, obs) for obs in observations]

    def params(self) -> dict[str, Any]:
        return {
            "agent_type": type(self).__name__,
            "agent_type_code": AGENT_NAME_CODES[type(self).__name__],
        }
