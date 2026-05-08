# pyright: reportPrivateImportUsage=false
"""Tests for supervised basic agent."""

import numpy as np
import torch

from figgie_gym.agent.supervised import (
    SimplifiedExpectedValueGeometricAggressive,
    SupervisedModelAgent,
)
from figgie_gym.envs.common import SUITS, ActionOnSuit, ObsOnSuit, ObsType
from figgie_gym.market.common import (
    OneSideQuote,
    Price,
    Quantity,
    Quote,
    TradeSummary,
)


def test_supervised_agent_act() -> None:
    # Mock model
    class MockModel(torch.nn.Module):
        def forward(self, _) -> torch.Tensor:  # noqa: ANN001
            # return some logits for which softmax will give something reasonable
            return torch.tensor([[4.0, 3.0, 2.0, 1.0]], dtype=torch.float32)

    ev_calc = SimplifiedExpectedValueGeometricAggressive(
        full_pot=400,
        goal_symbol_value=10,
        num_symbols=4,
        late_aggressiveness_factor=1.1,
    )

    agent = SupervisedModelAgent(
        model=MockModel(),
        ev_calculator=ev_calc,
        quote_spread=2,
        apply_soft_clip=True,
        preprocessor="nested",
    )

    # Create dummy observation
    def mk_obs_suit() -> ObsOnSuit:
        return ObsOnSuit(
            market_quote=Quote(
                bid=OneSideQuote(Price(5), Quantity(1)),
                ask=OneSideQuote(Price(6), Quantity(1)),
            ),
            last_price=Price(5),
            volume=Quantity(10),
            self_position=Quantity(2),
            known_count=Quantity(6),
            self_trade_summary=TradeSummary.new(),
            other_trade_summaries=[TradeSummary.new() for _ in range(4)],
        )

    suit_obs = {s: mk_obs_suit() for s in SUITS}
    observation = ObsType(10, 90, 0.9, 1000.0, suit_obs)

    rng = np.random.default_rng(42)
    belief, action, _ = agent.act(rng, observation)

    assert isinstance(belief, dict)
    assert len(belief) == 4

    assert isinstance(action, dict)
    assert len(action) == 4
    for s in SUITS:
        assert isinstance(action[s], ActionOnSuit)
        # Bid should be less than ask
        assert action[s].quote_bid <= action[s].quote_ask
