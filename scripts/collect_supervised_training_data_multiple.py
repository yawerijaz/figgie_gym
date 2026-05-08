from __future__ import annotations

from dataclasses import dataclass
from itertools import batched
from pathlib import Path, PosixPath, PurePosixPath
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import polars as pl
import torch
import tyro
from joblib import Parallel, delayed
from structlog import get_logger
from tensordict import (  # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs]
    TensorDict,
    cat,  # pyright: ignore[reportUnknownVariableType]
    make_tensordict,  # pyright: ignore[reportUnknownVariableType]
)
from tqdm import tqdm

from figgie_gym.agent.cardcounter import (
    CardCounterAgent,
    ExpectedValueGeometricAggressive,
    PosteriorProbabilies,
    get_symbol_quantity_permutations,
)
from figgie_gym.agent.noise import NoiseAgent
from figgie_gym.agent.supervised import (
    SimplifiedExpectedValueGeometricAggressive,
    SupervisedModelAgent,
)
from figgie_gym.envs.multiple_game import (
    AgentPool,
    AgentPoolMaker,
    MultipleGameRunner,
)
from figgie_gym.models.equivariant import EquivariantClassifier
from figgie_gym.models.fc_tensordict_classifier import FCTensorDictClassifier
from figgie_gym.models.naive_classifier import NaiveClassifier
from figgie_gym.models.suit_agent_equiv_classifier import (
    SuitAgentEquivClassifier,
)

if TYPE_CHECKING:
    from figgie_gym.pipelines.tendordict_preprocess import PreProcessorType

logger = get_logger(__name__)


@dataclass
class ModelSpec:
    checkpoint_path: str
    model_class_name: str
    preprocessor: PreProcessorType

    def load_model(self) -> torch.nn.Module:
        with torch.serialization.safe_globals(
            [PosixPath, Path, PurePosixPath],
        ):
            return class_map[self.model_class_name].load_from_checkpoint(  # pyright: ignore[reportUnknownMemberType]
                self.checkpoint_path,
            )


class_map = {
    model_class.__name__: model_class
    for model_class in [
        SuitAgentEquivClassifier,
        EquivariantClassifier,
        FCTensorDictClassifier,
    ]
}

model_specs = {
    "combinatoric": ModelSpec(
        checkpoint_path="/Users/yawerijaz/mlflow_dir/mlartifacts/6/97ce49d57d424e2d9e2231db646e8225/artifacts/epoch=27-step=44856/epoch=27-step=44856.ckpt",
        model_class_name="FCTensorDictClassifier",
        preprocessor="known_count_flat",
    ),
    "equivariant": ModelSpec(
        checkpoint_path="/Users/yawerijaz/mlflow_dir/mlartifacts/6/5ff7370a564e44b59bf9fbd5811f7fe1/artifacts/epoch=24-step=40050/epoch=24-step=40050.ckpt",
        model_class_name="EquivariantClassifier",
        preprocessor="nested",
    ),
    "suit_equivariant": ModelSpec(
        checkpoint_path="/Users/yawerijaz/mlflow_dir/mlartifacts/2/dbc0bfcb661c4cb2a9e5f5fd3748f05e/artifacts/epoch=3-step=6408/epoch=3-step=6408.ckpt",
        model_class_name="SuitAgentEquivClassifier",
        preprocessor="nested",
    ),
}


def make_agent_pool(rng: np.random.Generator) -> AgentPool:
    models_and_preprocessors = [
        (
            model_spec.load_model(),
            model_spec.preprocessor,
        )
        for model_spec in model_specs.values()
    ] + [(NaiveClassifier(), cast("PreProcessorType", "nested"))]
    return (
        [(NoiseAgent(), 0.3)]
        + [
            (
                CardCounterAgent(
                    ExpectedValueGeometricAggressive(
                        200,
                        10,
                        4,
                        rng.uniform(1, 10),
                        PosteriorProbabilies(
                            get_symbol_quantity_permutations(),
                        ),
                    ),
                    quote_spread=int(rng.integers(3, 10)),
                ),
                0.1 / 5,
            )
            for _ in range(5)
        ]
        + [
            (
                SupervisedModelAgent(
                    model=model,
                    ev_calculator=SimplifiedExpectedValueGeometricAggressive(
                        full_pot=400,
                        goal_symbol_value=10,
                        num_symbols=4,
                        late_aggressiveness_factor=rng.uniform(1, 10),
                    ),
                    quote_spread=int(rng.integers(3, 10)),
                    apply_soft_clip=True,
                    preprocessor=preprocessor,  # pyright: ignore[reportArgumentType]
                ),
                0.6 / 5 / len(models_and_preprocessors),
            )
            for model, preprocessor in models_and_preprocessors
            for _ in range(5)
        ]
    )


@dataclass
class Args:
    master_seed: int = 42
    num_runner: int = 200
    num_experiments_per_runner: int = 50
    num_steps: int = 40
    num_agents: int = 5
    output_dir: Path = Path("data/multi")
    output_dir_dev: Path = Path("data/multi_dev")
    dev: bool = False


def start_runner(
    runner: MultipleGameRunner,
    rng: np.random.Generator,
    agent_pool_maker: AgentPoolMaker,
    output_dir: Path,
) -> None:
    runner.construct_env_and_assign_agents(rng, agent_pool_maker)
    agent_pool_summary = pd.DataFrame(
        runner.agent_pool_summary(),
    )
    agent_pool_summary.to_parquet(
        output_dir
        / "agent_pool"
        / f"runner_{runner.game_runner_id:05g}.parquet",
    )

    game_run_logs = pd.DataFrame(
        list(runner.flattened_env_agent_snapshot_iter()),
    )
    # Remove null market bids and asks, non nulls are bid.price.value or ask.quantity.value
    game_run_logs = (
        game_run_logs.loc[
            :,
            ~(
                game_run_logs.columns.str.endswith("market_quote.bid.value")
                | game_run_logs.columns.str.endswith(
                    "market_quote.ask.value",
                )
            ),
        ]
        .rename(columns=lambda col: col.removesuffix(".value"))
        .sort_values(
            [
                "game_runner_id",
                "env_id",
                "game_runner_step",
                "agent_info.agent_id",
            ],
        )
    )
    game_run_logs.to_parquet(
        output_dir
        / "game_runs"
        / f"runner_{runner.game_runner_id:05g}.parquet",
    )

    game_summary_cols = [
        "game_runner_id",
        "env_id",
        "agent_info.agent_id",
    ]
    terminal_value = (
        game_run_logs[game_run_logs["next_observations.remaining_time"] == 0]
        .set_index(game_summary_cols)["rewards"]
        .to_frame("terminal_value")
        .reset_index()
    )

    game_run_logs = game_run_logs.merge(
        terminal_value.reset_index(),
        on=game_summary_cols,
    )
    game_run_logs.to_parquet(
        output_dir
        / "game_runs_with_terminal"
        / f"runner_{runner.game_runner_id:05g}.parquet",
    )


def make_tensordict_from_file_batch(file_paths: list[Path]) -> TensorDict:
    df = pl.read_parquet(file_paths, use_pyarrow=True)
    dict_of_list = cast(
        "dict[str, list[int | float]]",
        df.select(~pl.selectors.string()).to_pandas().to_dict(orient="list"),
    )  # pl.DataFrame.to_dict is slow, so convert to pandas first
    return make_tensordict(
        dict_of_list,
        batch_size=[len(df)],
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.dev:
        num_runner = 2
        num_experiments_per_runner = 50
        output_dir = args.output_dir_dev
    else:
        num_runner = args.num_runner
        num_experiments_per_runner = args.num_experiments_per_runner
        output_dir = args.output_dir

    master_rng = np.random.default_rng(args.master_seed)
    runner_rngs = master_rng.spawn(num_runner)
    runners = [
        (
            MultipleGameRunner(
                num_experiments_per_runner,
                args.num_steps,
                args.num_agents,
                game_runner_id=runner_id,
            ),
            rng,
        )
        for runner_id, rng in enumerate(runner_rngs)
    ]
    for subdirectory in [
        "agent_pool",
        "game_runs",
        "game_runs_with_terminal",
        "tensordict",
    ]:
        (output_dir / subdirectory).mkdir(parents=True, exist_ok=True)

    logger.info("Starting multi-game runners")

    Parallel(-1)(
        delayed(start_runner)(runner, rng, make_agent_pool, output_dir)
        for runner, rng in tqdm(runners, desc="Starting multi-game runners")
    )
    logger.info("Done with multi-game runners, starting tensordict creation")

    path_batches = tqdm(
        list(
            map(
                list,
                batched(
                    sorted((output_dir / "game_runs").glob("*")),
                    10,
                    strict=False,
                ),
            ),
        ),
        desc="Creating tensordict batches",
    )

    logger.info("Starting tensordict creation")

    result = cat(
        Parallel(-1)(
            delayed(make_tensordict_from_file_batch)(file_paths)
            for file_paths in path_batches
        ),
    )
    logger.info("Done with tensordict creation, writing to disk")
    result.memmap(str(output_dir / "tensordict"))
