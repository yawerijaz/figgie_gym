from __future__ import annotations

import contextlib
import webbrowser
from collections import Counter
from math import ceil, floor
from pathlib import Path
from tempfile import gettempdir
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding  # pyright: ignore[reportMissingTypeStubs]
from pettingzoo.utils.env import (  # pyright: ignore[reportMissingTypeStubs]
    ParallelEnv,
)

from figgie_gym.envs.common import (
    SUIT_PARTNER_MAP,
    SUITS,
    ActionType,
    AgentID,
    InfoType,
    ObsOnSuit,
    ObsType,
    RewardType,
    Suit,
)
from figgie_gym.envs.exception import EnvHasNotBeenResetError
from figgie_gym.market.common import (
    OrderType,
    Price,
    Quantity,
    Side,
    Symbol,
    TradeSummary,
)
from figgie_gym.market.exception import (
    PotentialShortSellError,
)
from figgie_gym.market.market import Market
from figgie_gym.market.market_display import MarketDisplay

if TYPE_CHECKING:
    from figgie_gym.market.client import MarketClient


class ParallelFiggieEnv(ParallelEnv[AgentID, ObsType, ActionType]):
    """A very small ParallelEnv skeleton for the Figgie trading game.

    This provides a starting point for integrating agents with the existing
    matching engine. It intentionally keeps the action/observation spaces small
    and implementation minimal so we can iterate on simultaneous-action logic
    later.

    Notes / limitations:
    - The underlying matching engine expects individual order submissions; the
      current step implementation submits each agent's order sequentially. This
      is a reasonable first step but not a perfect simultaneous-action model.
    - Rewards are a simple placeholder (zero); later we'll compute PnL or
      other game-specific rewards.
    """

    _market: Market
    _clients: dict[AgentID, MarketClient]
    render_filepathdir: Path

    def __init__(
        self,
        num_agents: int = 2,
        num_steps: int = 1000,
        render_mode: Literal["human"] | None = None,
        env_id: int | None = None,
    ) -> None:
        self.env_id = env_id
        self.metadata: dict[str, Any] = {"render.modes": ["human"]}
        self.num_steps = num_steps
        # If user didn't provide a render filepath, write HTML to a temp location
        # under /tmp/figgie so we don't clutter the repo.
        self.render_mode = render_mode
        self.render_filepathroot = Path(gettempdir()) / "figgie"

        # public config
        # use a private attribute name to avoid colliding with any base-class
        # properties (pettingzoo's ParallelEnv may define attributes named
        # similarly).
        self._num_agents = int(num_agents)
        self.symbols = SUITS

        # agent ids
        self.possible_agents: list[AgentID] = list(range(self._num_agents))
        self.agents: list[AgentID] = []
        self.is_env_ready = False

    def observation_space(self, agent: AgentID) -> spaces.Tuple:  # noqa: ARG002
        # Build an observation space matching ObsType: (cash, {suit: ObsOnSuit})
        # cash: scalar float
        cash_space = spaces.Box(
            low=-1e12,
            high=1e12,
            shape=(),
            dtype=np.float64,
        )

        # per-suit ObsOnSuit -> represent as a Dict:
        # market_quote -> flattened as bid_price, bid_quantity, ask_price, ask_quantity
        # last_price -> scalar float (use -1 for missing)
        # volume -> scalar int
        # self_position -> scalar int
        # self_trade_summary -> vector [buy_q, buy_consideration, sell_q, sell_consideration]
        # other_trade_summaries -> Sequence of the same vector
        per_suit_space = {}
        trade_summary_vec = spaces.Box(
            low=-1e12,
            high=1e12,
            shape=(4,),
            dtype=np.int64,
        )
        other_summaries_space = spaces.Sequence(trade_summary_vec)

        # WARN: the strucure is FLAT, unlink ObsType.
        for suit in self.symbols:
            per_suit_space[str(suit)] = spaces.Dict(
                {
                    "market_bid_price": spaces.Box(
                        low=-1.0,
                        high=1e9,
                        shape=(),
                        dtype=np.float64,
                    ),
                    "market_bid_quantity": spaces.Box(
                        low=0.0,
                        high=1e12,
                        shape=(),
                        dtype=np.float64,
                    ),
                    "market_ask_price": spaces.Box(
                        low=-1.0,
                        high=1e9,
                        shape=(),
                        dtype=np.float64,
                    ),
                    "market_ask_quantity": spaces.Box(
                        low=0.0,
                        high=1e12,
                        shape=(),
                        dtype=np.float64,
                    ),
                    "last_price": spaces.Box(
                        low=-1.0,
                        high=1e9,
                        shape=(),
                        dtype=np.float64,
                    ),
                    "volume": spaces.Box(
                        low=0,
                        high=1e12,
                        shape=(),
                        dtype=np.int64,
                    ),
                    "self_position": spaces.Box(
                        low=-1e12,
                        high=1e12,
                        shape=(),
                        dtype=np.int64,
                    ),
                    "self_trade_summary": trade_summary_vec,
                    "other_trade_summaries": other_summaries_space,
                },
            )

        suits_space = spaces.Dict(per_suit_space)

        # return a Tuple to reflect (cash, dict)
        return spaces.Tuple((cash_space, suits_space))

    def action_space(self, agent: AgentID) -> spaces.Space[object]:  # noqa: ARG002
        # ActionType is dict[Suit, ActionOnSuit], so create a Dict mapping each
        # suit name to a subspace with the four float fields.
        per_suit_action = {}
        for suit in self.symbols:
            per_suit_action[str(suit)] = spaces.Dict(
                {
                    "quote_bid": spaces.Box(
                        low=0.0,
                        high=1e9,
                        shape=(),
                        dtype=np.float64,
                    ),
                    "quote_ask": spaces.Box(
                        low=0.0,
                        high=1e9,
                        shape=(),
                        dtype=np.float64,
                    ),
                    "snipe_bid": spaces.Box(
                        low=0.0,
                        high=1e9,
                        shape=(),
                        dtype=np.float64,
                    ),
                    "snipe_ask": spaces.Box(
                        low=0.0,
                        high=1e9,
                        shape=(),
                        dtype=np.float64,
                    ),
                },
            )

        return spaces.Dict(per_suit_action)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, InfoType]]:
        if options and "rng" in options:
            if not isinstance(options["rng"], np.random.Generator):
                msg = (
                    f"must be np.random.Generator, got {type(options["rng"])=}"
                )
                raise ValueError(msg)
            self.np_random = options["rng"]
        else:
            self.np_random, self.np_random_seed = seeding.np_random(seed)

        # create a fresh market and clients
        self.agents = self.possible_agents.copy()
        self._market = Market(self.symbols)
        self.render_filepathdir = self.render_filepathroot / str(
            self._market.market_id,
        )
        self.render_filepathdir.mkdir(parents=True, exist_ok=True)

        self.steps_remaining = self.num_steps
        self.step_number = 0
        self._clients = {}
        self.goal_suit, initial_positions = self.create_intial_positions()
        if self.render_mode == "human":
            self.market_display = MarketDisplay(
                self._market,
                self._market.create_trade_ticker_tape(),
                self._market.create_order_ticker_tape(),
                "",
            )
            self.market_display.goal_suit = self.goal_suit
        for aid, pos in zip(
            self.possible_agents,
            initial_positions,
            strict=True,
        ):
            # Market.new_client returns a MarketClient which subscribes itself.
            client = self._market.new_client(
                client_id=aid,
                initial_positions=dict.fromkeys(self.symbols, Quantity(0))
                | pos,
                initial_cash=0,
            )
            self._clients[aid] = client

        self.is_env_ready = True
        observations = {agent: self._make_obs(agent) for agent in self.agents}

        return observations, dict[AgentID, InfoType]()

    def step(
        self,
        actions: dict[AgentID, ActionType],
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, RewardType],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, InfoType],
    ]:
        """Accept simultaneous actions (dict agent->action) and apply them.

        Each submitted order is sent to the market in a random sequence.
        """
        if not self.is_env_ready:
            raise EnvHasNotBeenResetError

        rewards = dict.fromkeys(self.agents, 0.0)
        truncations = dict.fromkeys(self.agents, False)
        infos = dict[AgentID, InfoType].fromkeys(
            self.agents,
            dict[str, str](),
        )
        if self.steps_remaining < 0:
            observations = {
                agent: self._make_obs(agent) for agent in self.agents
            }
            terminations = dict.fromkeys(self.agents, True)
            return observations, rewards, terminations, truncations, infos

        self._market.advance_clock()
        agent_actions = list(actions.items())

        # MAYBE consider have different random order on each symbol
        # apply all agents' "snipe" actions after shuffling
        self.np_random.shuffle(agent_actions)
        for agent, action in agent_actions:
            if agent not in self.agents:
                continue
            self._clients[agent].cancel_all_active_orders()
            for suit, action_on_suit in action.items():
                self._clients[agent].send_order(
                    Symbol(suit),
                    Side.BUY,
                    Price(floor(action_on_suit.snipe_bid)),
                    order_type=OrderType.IMMEDIATE_OR_CANCEL,
                )
                with contextlib.suppress(PotentialShortSellError):
                    self._clients[agent].send_order(
                        Symbol(suit),
                        Side.SELL,
                        Price(ceil(action_on_suit.snipe_ask)),
                        order_type=OrderType.IMMEDIATE_OR_CANCEL,
                    )

                self._clients[agent].send_order(
                    Symbol(suit),
                    Side.BUY,
                    Price(floor(action_on_suit.quote_bid)),
                )
                with contextlib.suppress(PotentialShortSellError):
                    self._clients[agent].send_order(
                        Symbol(suit),
                        Side.SELL,
                        Price(ceil(action_on_suit.quote_ask)),
                    )

        # build observations
        observations = {agent: self._make_obs(agent) for agent in self.agents}

        if self.steps_remaining <= 0:
            # Distribute final bonus and card values
            goal_suit_holdings = {
                ag: observations[ag].per_suit[self.goal_suit].self_position
                for ag in self.agents
            }
            # Recognize cash as part of the reward at the end of the game,
            # as we don't have a good reward signal during the game yet.
            # Doing otherwise may encourage premature liquidation.
            # Reward shaping is done outside of the env for better control.
            cash_holdings = {ag: observations[ag].cash for ag in self.agents}
            goal_max = max(goal_suit_holdings.values())
            total_bonus = 200 - 10 * sum(goal_suit_holdings.values())
            awardees = [
                ag for ag, q in goal_suit_holdings.items() if q == goal_max
            ]
            bonus_per_awardee = total_bonus / len(awardees)
            rewards = {
                ag: q * 10.0
                + (bonus_per_awardee if ag in awardees else 0)
                + cash_holdings[ag]
                for ag, q in goal_suit_holdings.items()
            }

        terminations = dict.fromkeys(self.agents, self.steps_remaining <= 0)

        self.steps_remaining -= 1
        self.step_number += 1

        return observations, rewards, terminations, truncations, infos

    def render(self) -> None:
        if self.render_mode is None:
            return
        path = self.market_display.print_html(
            filepath=self.render_filepathdir / f"{self._market.time:05g}.html",
        )
        if self.steps_remaining <= 0:
            full_path = path + f"?minstep=1&maxstep={self.num_steps}"
            redirect_url = self.market_display.print_redirect_page(
                full_path,
                self.render_filepathdir / "redirect.html",
            )
            webbrowser.open(redirect_url.as_uri())

    def close(self) -> None:
        self._clients = {}
        self.agents = []

    def _make_obs(self, agent: AgentID) -> ObsType:
        if not self.is_env_ready:
            raise EnvHasNotBeenResetError

        client = self._clients.get(agent)
        if client is None:
            msg = f"Unknown agent {agent}"
            raise KeyError(msg)

        # Cash: use MarketClient.cash property (initial_cash + net movements)
        cash: float = float(client.cash)

        obs_by_suit: dict[Suit, ObsOnSuit] = {}

        summarizer = self._market.trade_summary
        spotter = self._market.spotter

        for suit in self.symbols:
            # market quote for this suit (spotter provides a default Quote)
            market_quote = spotter.spots[suit]

            # last price and volume from trade summarizer (use safe defaults)
            last_price = summarizer.last_prices[suit]

            volume = summarizer.volumes[suit]

            # self position for this suit
            self_position = client.order_managers[suit].quantity

            # trade summaries: own and others
            self_trade_summary = summarizer.summary[suit][client]
            other_trade_summaries: list[TradeSummary] = [
                summarizer.summary[suit][other_client]
                for other_client in self._clients.values()
                if other_client is not client
            ]
            others_position = sum(
                o.buy_quantity - o.sell_quantity - o.min_net_quantity_change
                for o in other_trade_summaries
            )
            obs_by_suit[suit] = ObsOnSuit(
                market_quote,
                last_price,
                volume,
                self_position,
                Quantity(self_position + others_position),
                self_trade_summary,
                other_trade_summaries,
            )

        return ObsType(
            self.step_number,
            self.steps_remaining,
            self.steps_remaining / (self.step_number + self.steps_remaining),
            cash,
            obs_by_suit,
        )

    def create_intial_positions(
        self,
    ) -> tuple[Symbol, list[dict[Symbol, Quantity]]]:
        suits = self.np_random.permutation(self.symbols)
        twelve_suit, eight_suit, ten_suit1, ten_suit2 = cast(
            "tuple[Symbol, Symbol, Symbol, Symbol]",
            suits,
        )
        goal_suit = SUIT_PARTNER_MAP[twelve_suit]

        asset_per_agents = 40 // self.num_agents

        cards = (
            [twelve_suit] * 12
            + [eight_suit] * 8
            + [ten_suit1] * 10
            + [ten_suit2] * 10
        )
        self.np_random.shuffle(cards)

        starting_positions = [
            {
                k: Quantity(v)
                for k, v in Counter(
                    cards[asset_per_agents * i : asset_per_agents * (i + 1)],
                ).items()
            }
            for i in range(self.num_agents)
        ]
        return goal_suit, starting_positions
