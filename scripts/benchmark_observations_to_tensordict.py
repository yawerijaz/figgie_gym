"""Benchmark different ways to convert a list[ObsType] into a TensorDict.

Methods:
 - observations_to_tensordict (from observations_to_tensordict.py)
 - observations_to_tensordict_pandas (from observations_to_tensordict.py)
 - manual_dict_of_lists (replicates the manual building used in SupervisedModelAgent)

Usage:
    uv run python scripts/benchmark_observations_to_tensordict.py

"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from statistics import mean

from dataclasses import asdict
from tensordict import TensorDict, make_tensordict

from figgie_gym.envs.common import SUITS, ObsOnSuit, ObsType
from figgie_gym.market.common import Quantity, Quote, TradeSummary
from figgie_gym.pipelines.observations_to_tensordict import (
    observations_to_tensordict,
    observations_to_tensordict_pandas,
)
from figgie_gym.utilities import flatten


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


def manual_dict_of_lists(observations: list[ObsType]) -> TensorDict:
    # Build dict[str, list[float]] similar to SupervisedModelAgent (fast path)
    agg: dict[str, list[float]] = defaultdict(list)
    for o in observations:
        agg["current_observations.time"].append(float(o.time))
        agg["current_observations.remaining_time"].append(
            float(o.remaining_time)
        )
        agg["current_observations.remaining_time_fraction"].append(
            float(o.remaining_time_fraction),
        )
        agg["current_observations.cash"].append(float(o.cash))
        for sym in SUITS:
            obs = o.per_suit[sym]
            prefix = f"current_observations.per_suit.{sym}"
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
            agg[f"{prefix}.last_price"].append(
                float("nan")
                if obs.last_price is None
                else float(obs.last_price),
            )
            agg[f"{prefix}.volume"].append(float(obs.volume))
            agg[f"{prefix}.self_position"].append(float(obs.self_position))
            agg[f"{prefix}.known_count"].append(float(obs.known_count))

            st = obs.self_trade_summary
            agg[f"{prefix}.self_trade_summary.buy_quantity"].append(
                float(st.buy_quantity)
            )
            agg[f"{prefix}.self_trade_summary.buy_consideration"].append(
                float(st.buy_consideration),
            )
            agg[f"{prefix}.self_trade_summary.sell_quantity"].append(
                float(st.sell_quantity)
            )
            agg[f"{prefix}.self_trade_summary.sell_consideration"].append(
                float(st.sell_consideration),
            )
            agg[f"{prefix}.self_trade_summary.min_net_quantity_change"].append(
                float(st.min_net_quantity_change),
            )

            for i, oth in enumerate(obs.other_trade_summaries):
                base = f"{prefix}.other_trade_summaries.item_{i}"
                agg[f"{base}.buy_quantity"].append(float(oth.buy_quantity))
                agg[f"{base}.buy_consideration"].append(
                    float(oth.buy_consideration)
                )
                agg[f"{base}.sell_quantity"].append(float(oth.sell_quantity))
                agg[f"{base}.sell_consideration"].append(
                    float(oth.sell_consideration)
                )
                agg[f"{base}.min_net_quantity_change"].append(
                    float(oth.min_net_quantity_change)
                )

    return make_tensordict(dict(agg), batch_size=[len(observations)])

def manual_dict_of_lists_flatten(observations: list[ObsType]) -> TensorDict:
    flatteneds = [flatten(asdict(o))for o in observations]
    keys = flatteneds[0].keys()
    agg: dict[str, list[float]] = defaultdict(list)
    for o in flatteneds:
        for k in keys:
            v = o[k]
            agg[k].append(float('nan') if v is None else float(v))
    return make_tensordict(dict(agg), batch_size=[len(observations)])


def timeit(
    fn: Callable[[list[ObsType]], TensorDict],
    observations: list[ObsType],
    repeat=3,
):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        td = fn(observations)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    # return mean time and a quick correctness check shape
    return mean(times), td


def run():
    sizes = [1, 10, 100, 1000, 5000]
    methods = [
        ("observations_to_tensordict", observations_to_tensordict),
        (
            "observations_to_tensordict_pandas",
            observations_to_tensordict_pandas,
        ),
        ("manual_dict_of_lists", manual_dict_of_lists),
        ("manual_dict_of_lists_flatten", manual_dict_of_lists_flatten),
    ]

    print("Benchmark: converting list[ObsType] -> TensorDict")
    for n in sizes:
        observations = [make_obs() for _ in range(n)]
        print(f"\nN={n}")
        for name, fn in methods:
            t, td = timeit(fn, observations, repeat=3)
            print(f"  {name:30s}: {t:.6f}s  -> td.shape={td.shape}")


if __name__ == "__main__":
    run()
