from typing import cast

from tensordict import (  # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs]
    TensorDict,
    stack,  # pyright: ignore[reportUnknownVariableType]
)

from figgie_gym.envs.common import SUITS, ObsOnSuit, ObsType
from figgie_gym.market.common import Quantity, Quote, TradeSummary
from figgie_gym.models.equivariant import EquivariantObservationArchitectureArgs
from figgie_gym.models.mappo import CentralizedCriticNet
from figgie_gym.pipelines.observations_to_tensordict import (
    observations_to_tensordict_direct,
)
from figgie_gym.pipelines.tendordict_preprocess import preprocessors
from figgie_gym.utilities import get_device


def _make_obs() -> ObsType:
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
                self_trade_summary=TradeSummary(
                    buy_quantity=Quantity(0),
                    buy_consideration=0,
                    sell_quantity=Quantity(0),
                    sell_consideration=0,
                    min_net_quantity_change=Quantity(0),
                ),
                other_trade_summaries=[
                    TradeSummary(
                        buy_quantity=Quantity(0),
                        buy_consideration=0,
                        sell_quantity=Quantity(0),
                        sell_consideration=0,
                        min_net_quantity_change=Quantity(0),
                    ),
                ]
                * 4,
            )
            for s in SUITS
        },
    )


def test_centralized_critic_output_shape() -> None:
    equiv_args = EquivariantObservationArchitectureArgs(
        trade_summary_input_dim=5,
        trade_summary_hidden_dims=(32,),
        trade_summary_embed_dim=16,
        known_trade_aggregate_hidden_dims=(32,),
        known_trade_aggregate_embed_dim=12,
        private_holding_in_dim=8,
        suit_embed_hidden_dims=(32,),
        suit_embed_dim=16,
        other_suit_context_hidden_dims=(32,),
    )
    dev = get_device()
    critic = CentralizedCriticNet(
        equiv_args,
        num_learned_agents=2,
        fusion_hidden_dims=(16,),
    ).to(dev)

    batch_n = 4
    obs_a = [_make_obs() for _ in range(batch_n)]
    obs_b = [_make_obs() for _ in range(batch_n)]

    def _x(obs_list: list[ObsType]) -> TensorDict:
        td = observations_to_tensordict_direct(obs_list)
        proc = preprocessors["nested"].preprocess(
            td.unflatten_keys(),
            apply_soft_clip_on_prices=True,
            stage="inference",
        )
        return cast("TensorDict", proc["x"].to(dev))  # pyright: ignore[reportUnknownMemberType]

    joint = stack([_x(obs_a), _x(obs_b)], dim=1)
    out = critic(joint)
    assert tuple(out.shape) == (batch_n, 2)
    assert out.dtype.is_floating_point
