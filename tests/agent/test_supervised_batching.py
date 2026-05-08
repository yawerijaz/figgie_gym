# pyright: reportPrivateImportUsage=false
from typing import Any

import numpy as np
import torch

from figgie_gym.agent.supervised import (
    SimplifiedExpectedValueGeometricAggressive,
    SupervisedModelAgent,
)
from figgie_gym.envs.common import SUITS, ObsOnSuit, ObsType
from figgie_gym.market.common import (
    Price,
    Quantity,
    Quote,
    Symbol,
    TradeSummary,
)


def test_supervised_batch_inference() -> None:
    # Setup model and agent
    # We'll mock the model output for consistency
    class MockModel(torch.nn.Module):
        def forward(self, x: Any) -> torch.Tensor:  # noqa: ANN401
            bs = x.batch_size[0]
            return torch.zeros((bs, 4))

    ev = SimplifiedExpectedValueGeometricAggressive(400, 10, 4, 1.5)
    model = MockModel()
    agent = SupervisedModelAgent(
        model,
        ev,
        quote_spread=5,
        apply_soft_clip=True,
        preprocessor="nested",
    )

    # Create dummy observations
    def make_dummy_obs(step: int) -> ObsType:
        suit_obs = dict[Symbol, ObsOnSuit]()
        for s in SUITS:
            suit_obs[s] = ObsOnSuit(
                market_quote=Quote(None, None),
                last_price=None,
                volume=Quantity(0),
                self_position=Quantity(2),
                known_count=Quantity(8),
                self_trade_summary=TradeSummary(
                    Quantity(0),
                    Price(0),
                    Quantity(0),
                    Price(0),
                    Quantity(0),
                ),
                other_trade_summaries=[
                    TradeSummary(
                        Quantity(0),
                        Price(0),
                        Quantity(0),
                        Price(0),
                        Quantity(0),
                    ),
                ],
            )
        return ObsType(step, 100 - step, 1 - step / 100, 1000.0, suit_obs)

    obs1 = make_dummy_obs(10)
    obs2 = make_dummy_obs(20)
    observations: list[ObsType] = [obs1, obs2]

    # Batch prediction
    batch_probs = agent.predict_probs(observations)
    assert batch_probs.shape == (2, 4)

    # Individual predictions
    prob1 = agent.predict_probs([obs1])[0]
    prob2 = agent.predict_probs([obs2])[0]

    np.testing.assert_allclose(batch_probs[0], prob1)
    np.testing.assert_allclose(batch_probs[1], prob2)

    # Compute actions
    action1 = agent.compute_action_from_probs(prob1, obs1)
    action2 = agent.compute_action_from_probs(prob2, obs2)

    batch_action1 = agent.compute_action_from_probs(batch_probs[0], obs1)
    batch_action2 = agent.compute_action_from_probs(batch_probs[1], obs2)

    assert action1 == batch_action1
    assert action2 == batch_action2
