from dataclasses import asdict
from typing import Any

import pandas as pd
from tensordict import (  # pyright: ignore[reportMissingTypeStubs]
    TensorDict,
    make_tensordict,  # pyright: ignore[reportUnknownVariableType]
)

from figgie_gym.envs.common import SUITS, ObsType
from figgie_gym.utilities import flatten


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
    """Legacy wrapper for compatibility.

    This used to build per-observation small TensorDicts and concat them,
    which is inefficient. For speed and robustness we now delegate to
    :func:`observations_to_tensordict_direct` which constructs a single
    columnar dict-of-lists and is much faster at scale.
    """
    return observations_to_tensordict_direct(observations)


def observations_to_tensordict_pandas(
    observations: list[ObsType],
) -> TensorDict:
    clean_dicts = [
        {
            "current_observations." + k.removesuffix(".value"): (
                float("nan") if v is None else v
            )
            for k, v in obs_to_clean_dict(o).items()
        }
        for o in observations
    ]

    return make_tensordict(
        pd.DataFrame(clean_dicts).to_dict(orient="list"),  # pyright: ignore[reportArgumentType]
        batch_size=[len(clean_dicts)],
    ).float()


def observations_to_tensordict_direct(
    observations: list[ObsType],
) -> TensorDict:
    """Efficient, maintainable conversion from list[ObsType] to TensorDict.

    - Builds a dict-of-lists by crawling the ObsType structure directly (no
      per-observation flatten/asdict calls), preserving the key naming used by
      the other helpers (prefixed with `current_observations.`).
    - Converts None to float('nan') and ensures all values are floats.

    This avoids small TensorDict allocations and pandas overhead and is
    resilient to small changes because it enumerates fields in one place.
    """
    if not observations:
        return make_tensordict({}, batch_size=[0]).float()

    agg: dict[str, list[float]] = {}

    def ensure_key(k: str) -> None:
        if k not in agg:
            agg[k] = []

    for o in observations:
        # Top-level
        ensure_key("current_observations.time")
        agg["current_observations.time"].append(float(o.time))

        ensure_key("current_observations.remaining_time")
        agg["current_observations.remaining_time"].append(
            float(o.remaining_time),
        )

        ensure_key("current_observations.remaining_time_fraction")
        agg["current_observations.remaining_time_fraction"].append(
            float(o.remaining_time_fraction),
        )

        ensure_key("current_observations.cash")
        agg["current_observations.cash"].append(float(o.cash))

        for sym in SUITS:
            obs = o.per_suit[sym]
            prefix = f"current_observations.per_suit.{sym}"

            # market quote
            ensure_key(f"{prefix}.market_quote.bid.price")
            ensure_key(f"{prefix}.market_quote.bid.quantity")
            ensure_key(f"{prefix}.market_quote.ask.price")
            ensure_key(f"{prefix}.market_quote.ask.quantity")

            agg[f"{prefix}.market_quote.bid.price"].append(
                float("nan")
                if obs.market_quote.bid is None
                else float(obs.market_quote.bid.price),
            )
            agg[f"{prefix}.market_quote.bid.quantity"].append(
                float("nan")
                if obs.market_quote.bid is None
                else float(obs.market_quote.bid.quantity),
            )
            agg[f"{prefix}.market_quote.ask.price"].append(
                float("nan")
                if obs.market_quote.ask is None
                else float(obs.market_quote.ask.price),
            )
            agg[f"{prefix}.market_quote.ask.quantity"].append(
                float("nan")
                if obs.market_quote.ask is None
                else float(obs.market_quote.ask.quantity),
            )

            # simple fields
            ensure_key(f"{prefix}.last_price")
            ensure_key(f"{prefix}.volume")
            ensure_key(f"{prefix}.self_position")
            ensure_key(f"{prefix}.known_count")

            agg[f"{prefix}.last_price"].append(
                float("nan")
                if obs.last_price is None
                else float(obs.last_price),
            )
            agg[f"{prefix}.volume"].append(float(obs.volume))
            agg[f"{prefix}.self_position"].append(float(obs.self_position))
            agg[f"{prefix}.known_count"].append(float(obs.known_count))

            # self trade summary
            st = obs.self_trade_summary
            for field in [
                "buy_quantity",
                "buy_consideration",
                "sell_quantity",
                "sell_consideration",
                "min_net_quantity_change",
            ]:
                key = f"{prefix}.self_trade_summary.{field}"
                ensure_key(key)
                agg[key].append(float(getattr(st, field)))

            # other trade summaries
            for i, oth in enumerate(obs.other_trade_summaries):
                base = f"{prefix}.other_trade_summaries.item_{i}"
                for field in [
                    "buy_quantity",
                    "buy_consideration",
                    "sell_quantity",
                    "sell_consideration",
                    "min_net_quantity_change",
                ]:
                    key = f"{base}.{field}"
                    ensure_key(key)
                    agg[key].append(float(getattr(oth, field)))

    # make tensordict
    return make_tensordict(dict(agg), batch_size=[len(observations)]).float()
