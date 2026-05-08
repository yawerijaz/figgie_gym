from tensordict import TensorDict  # pyright: ignore[reportMissingTypeStubs]

from figgie_gym.envs.common import SUITS, ObsOnSuit, ObsType
from figgie_gym.market.common import Quantity, Quote, TradeSummary
from figgie_gym.pipelines.observations_to_tensordict import (
    observations_to_tensordict,
    observations_to_tensordict_direct,
    observations_to_tensordict_pandas,
)


def make_obs() -> ObsType:
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
                self_trade_summary=TradeSummary(0, 0, 0, 0, 0),
                other_trade_summaries=[TradeSummary(0, 0, 0, 0, 0)] * 4,
            )
            for s in SUITS
        },
    )


def normalize_keys(td: TensorDict) -> list[str]:
    # ensure same set of keys listing
    return sorted(td.keys())  # pyright: ignore[reportReturnType, reportUnknownMemberType]


def test_equivalence_small() -> None:
    obs = [make_obs() for _ in range(5)]
    a = observations_to_tensordict(obs)
    b = observations_to_tensordict_pandas(obs)
    c = observations_to_tensordict_direct(obs)

    assert a.shape == b.shape == c.shape
    assert normalize_keys(a) == normalize_keys(b) == normalize_keys(c)


def test_equivalence_single() -> None:
    obs = [make_obs()]
    a = observations_to_tensordict(obs)
    b = observations_to_tensordict_pandas(obs)
    c = observations_to_tensordict_direct(obs)

    assert a.shape == b.shape == c.shape
    assert normalize_keys(a) == normalize_keys(b) == normalize_keys(c)
