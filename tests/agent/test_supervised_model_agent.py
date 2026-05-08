import numpy as np
import torch
from tensordict import TensorDict  # pyright: ignore[reportMissingTypeStubs]

from figgie_gym.agent.supervised import (
    SimplifiedExpectedValueGeometricAggressive,
    SupervisedModelAgent,
)
from figgie_gym.envs.common import SUITS, ObsOnSuit, ObsType
from figgie_gym.market.common import Quantity, Quote, TradeSummary


class DummyModel(torch.nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.num_classes = num_classes

    def forward(self, x: TensorDict) -> torch.Tensor:
        return torch.ones(len(x), self.num_classes)


def make_obs() -> ObsType:
    ts = TradeSummary(Quantity(0), 0, Quantity(0), 0, Quantity(0))
    return ObsType(
        10,
        10,
        0.5,
        100,
        {
            s: ObsOnSuit(
                market_quote=Quote(None, None),
                last_price=None,
                volume=Quantity(0),
                self_position=Quantity(2),
                known_count=Quantity(8),
                self_trade_summary=ts,
                other_trade_summaries=[ts] * 4,
            )
            for s in SUITS
        },
    )


def test_predict_probs_nested_preprocessor() -> None:
    model = DummyModel()
    agent = SupervisedModelAgent(
        model=model,
        ev_calculator=SimplifiedExpectedValueGeometricAggressive(
            400,
            10,
            4,
            late_aggressiveness_factor=2.0,
        ),
        quote_spread=5,
        apply_soft_clip=True,
        preprocessor="nested",
    )
    probs = agent.predict_probs([make_obs()])
    assert probs.shape == (1, 4)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_predict_probs_known_count_flat_preprocessor() -> None:
    model = DummyModel()
    agent = SupervisedModelAgent(
        model=model,
        ev_calculator=SimplifiedExpectedValueGeometricAggressive(
            400,
            10,
            4,
            late_aggressiveness_factor=2.0,
        ),
        quote_spread=5,
        apply_soft_clip=True,
        preprocessor="known_count_flat",
    )
    probs = agent.predict_probs([make_obs()])
    assert probs.shape == (1, 4)
    assert np.allclose(probs.sum(axis=1), 1.0)
