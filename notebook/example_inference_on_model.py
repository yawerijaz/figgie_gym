# %%
# %%
from pathlib import Path, PosixPath, PurePosixPath

import torch

from figgie_gym.models.equivariant import EquivariantClassifier
from figgie_gym.models.fc_tensordict_classifier import FCTensorDictClassifier
from figgie_gym.models.naive_classifier import NaiveClassifier
from figgie_gym.models.suit_agent_equiv_classifier import (
    SuitAgentEquivClassifier,
)

checkpoint_path = "/Users/yawerijaz/mlflow_dir/mlartifacts/2/dbc0bfcb661c4cb2a9e5f5fd3748f05e/artifacts/epoch=3-step=6408/epoch=3-step=6408.ckpt"
equiv_path = "/Users/yawerijaz/mlflow_dir/mlartifacts/6/5ff7370a564e44b59bf9fbd5811f7fe1/artifacts/epoch=24-step=40050/epoch=24-step=40050.ckpt"
comib_path = "/Users/yawerijaz/mlflow_dir/mlartifacts/6/97ce49d57d424e2d9e2231db646e8225/artifacts/epoch=27-step=44856/epoch=27-step=44856.ckpt"

with torch.serialization.safe_globals(
    [PosixPath, Path, PurePosixPath],
):
    model_sae = SuitAgentEquivClassifier.load_from_checkpoint(  # pyright: ignore[reportUnknownMemberType]
        checkpoint_path,
    )
    model_equiv = EquivariantClassifier.load_from_checkpoint(  # pyright: ignore[reportUnknownMemberType]
        equiv_path,
    )
    model_combi = FCTensorDictClassifier.load_from_checkpoint(  # pyright: ignore[reportUnknownMemberType]
        comib_path,
    )
    model_sae.eval()
    model_equiv.eval()
    model_combi.eval()
    model_naive = NaiveClassifier()

# %%
import numpy as np

from figgie_gym.agent.supervised import (
    SimplifiedExpectedValueGeometricAggressive,
    SupervisedModelAgent,
)

rng = np.random.default_rng(42)

m = SupervisedModelAgent(
    model=model_combi,
    ev_calculator=SimplifiedExpectedValueGeometricAggressive(
        full_pot=400,
        goal_symbol_value=10,
        num_symbols=4,
        late_aggressiveness_factor=rng.uniform(1, 10),
    ),
    quote_spread=int(rng.integers(3, 10)),
)

# %%
from dataclasses import asdict

from tensordict import TensorDict, stack

from figgie_gym.envs.common import SUITS, ObsOnSuit, ObsType
from figgie_gym.market.common import Price, Quantity, Quote, TradeSummary
from figgie_gym.pipelines.tendordict_preprocess import (
    preprocessors,
)
from figgie_gym.utilities import flatten

o = ObsType(
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
                )
            ]
            * 4,
        )
        for s in SUITS
    },
)

# %%
from typing import Any
from tensordict import make_tensordict

def obs_to_clean_dict(o: ObsType) -> dict[str, Any]:
    o_dict = flatten(asdict(o))
    for suit in SUITS:
        for way in ["bid", "ask"]:
            quote_prefix = f"per_suit.{suit}.market_quote.{way}"
            if f"{quote_prefix}.value" in o_dict:
                del o_dict[f"{quote_prefix}.value"]
                o_dict[f"{quote_prefix}.price.value"] = float("nan")
                o_dict[f"{quote_prefix}.quantity.value"] = float("nan")
    return o_dict

def observations_to_tensordict(observations: list[ObsType]) -> TensorDict:
    clean_dicts = [{
            "current_observations." + k.removesuffix(".value"): [
                (float("nan") if v is None else v)
            ]
            for k, v in obs_to_clean_dict(o).items()
        } for o in observations]

    tds = [make_tensordict(
        clean_dict,
        batch_size=[1],
    ) for clean_dict in clean_dicts]
    return TensorDict.cat(tds, dim=0).float()

model_equiv.to(device="mps")(preprocessors["nested"].preprocess(
    observations_to_tensordict([o, o, o]).unflatten_keys(), True, "inference"
)['x'].to(device='mps'))


# %%
preprocessed_td.float().mean().to_dict()
# preprocessed_td_make_td = preprocessed_td
# preprocessed_td_direct = preprocessed_td
(preprocessed_td_direct - preprocessed_td_make_td).to_dict()

# %%
from tensordict import make_tensordict

td = make_tensordict(
    {
        "current_observations." + k.removesuffix(".value"): [
            (float("nan") if v is None else v)
        ]
        * 2
        for k, v in obs_to_clean_dict(o).items()
    },
    batch_size=[2],
)
# td = td_direct
preprocessed_td = preprocessors["known_count_flat"].preprocess(
    td.unflatten_keys(), True, "inference"
)
print(preprocessed_td["x"].to(device="mps").float().to_dict())
model_combi.to(device="mps").forward(
    preprocessed_td["x"].to(device="mps").float()
)

# %%
from tensordict import make_tensordict

td = make_tensordict(
    {
        "current_observations." + k.removesuffix(".value"): [
            (float("nan") if v is None else v)
        ]
        * 2
        for k, v in obs_to_clean_dict(o).items()
    },
    batch_size=[2],
)
preprocessed_td = preprocessors["nested"].preprocess(
    td.unflatten_keys(), True, "inference"
)
model_equiv.to(device="mps").forward(
    preprocessed_td["x"].to(device="mps").float()
)

# %%


example_td = TensorDict.load_memmap("../../..//data/multi_dev/tensordict/")
