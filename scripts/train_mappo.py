import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from lightning.pytorch.loggers import MLFlowLogger
from tensordict import (  # pyright: ignore[reportMissingTypeStubs]
    TensorDict,  # pyright: ignore[reportUnknownVariableType]
    stack,  # pyright: ignore[reportUnknownVariableType]
)
from torch import nn

from figgie_gym.agent.cardcounter import (
    CardCounterAgent,
    ExpectedValueGeometricAggressive,
    PosteriorProbabilies,
    get_symbol_quantity_permutations,
)
from figgie_gym.agent.noise import (
    NoiseAgent,
)
from figgie_gym.agent.ppo_family import MAPPOAgent
from figgie_gym.envs.common import (
    SUIT_TO_CODE,
    BeliefType,
    Time,
)
from figgie_gym.envs.parallel_figgie_env import ParallelFiggieEnv
from figgie_gym.envs.vectorized_game_runner import (
    AgentPoolID,
    EnvAgentStepSnapshot,
    VectorizedGameRunner,
)
from figgie_gym.envs.vectorized_parallel_figgie_env import (
    VectorizedParallelFiggieEnv,
)
from figgie_gym.models.equivariant import (
    EquivariantObservationArchitectureArgs,
)
from figgie_gym.models.mappo import CentralizedCriticNet, MAPPONetworks
from figgie_gym.models.mappo_blocks import MAPPOLightningModule
from figgie_gym.models.ppo import (
    EquivariantActorNet,
    EquivariantPricerNet,
)
from figgie_gym.models.ppo_helpers import (
    action_adaptor,
    compute_adjusted_reward_from_snapshot,
)
from figgie_gym.pipelines.observations_to_tensordict import (
    observations_to_tensordict_direct,
)
from figgie_gym.pipelines.tendordict_preprocess import (
    preprocessors,
)
from figgie_gym.training.ppo_logging_utilities import (
    aggregate_beliefs_by_suit_role,
    log_clairvoyant_metrics,
    log_debug_metrics_for_iteration,
    log_reward_metrics,
)
from figgie_gym.utilities import get_device, get_git_sha

if TYPE_CHECKING:
    import mlflow


def train(  # noqa: C901, PLR0913, PLR0915
    max_steps_per_game: int,
    num_games_per_iteration: int,
    num_iterations: int,
    num_epochs_per_iteration: int,
    rng: np.random.Generator,
    description: str = "MAPPO training run",
    mlflow_experiment: str = "Figgie MAPPO",
) -> None:
    # Setup MLFlow logger (similar to scripts/train.py)
    artifact_dir = Path("mlartifacts/")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    mlflow_logger = MLFlowLogger(
        experiment_name=mlflow_experiment,
        tracking_uri=mlflow_uri,
        tags={
            "mlflow.note.content": description,
            "git_commit_sha": get_git_sha(),
        },
        log_model="all",
        artifact_location=str(artifact_dir.resolve()),
    )

    # Extract underlying mlflow client and run_id for use in custom loop
    mlflow_client = cast("mlflow.MlflowClient", mlflow_logger.experiment)  # pyright: ignore[reportUnknownMemberType]
    mlflow_run_id = cast("str", mlflow_logger._run_id)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    num_learned_agents = 2
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

    centralized_critic = CentralizedCriticNet(
        equiv_args,
        num_learned_agents=num_learned_agents,
        fusion_hidden_dims=(64,),
    ).to(get_device())

    actor_net = EquivariantActorNet(
        equiv_args,
        actor_hidden_dims=(32,),
        lr=1e-3,
    ).to(get_device())

    belief_net = EquivariantPricerNet(
        equiv_args,
        pricer_hidden_dims=(32,),
        lr=1e-3,
    ).to(get_device())

    networks = MAPPONetworks(
        centralized_critic=centralized_critic,
        actor=actor_net,
        belief=belief_net,
        belief_embedder=nn.Linear(1, 16).to(get_device()),
    )
    lightning_module = MAPPOLightningModule(
        mappo_networks=networks,
        num_learned_agents=num_learned_agents,
        ppo_epsilon=0.3,
        critic_loss_coef=0.5,
        actor_entropy_loss_coef=0.001,
        belief_loss_coef=5.0,
        learning_rate=0.0002,
    )
    mappo = MAPPOAgent(
        lightning_module=lightning_module,
        networks=networks,
    )
    card_counter = CardCounterAgent(
        ExpectedValueGeometricAggressive(
            200,
            10,
            4,
            5.0,
            PosteriorProbabilies(
                get_symbol_quantity_permutations(),
            ),
        ),
        quote_spread=4,
    )
    noise = NoiseAgent()
    # "noise" noise
    # "card_counter" card_counter_agent
    agent_pool = [
        card_counter,
        card_counter,
        noise,
        mappo,
        mappo,
    ]

    agent_pool_agent_types = [
        "card_counter",
        "card_counter",
        "noise",
        "mappo",
        "mappo",
    ]
    mappo_pool_indices = [
        i for i, t in enumerate(agent_pool_agent_types) if t == "mappo"
    ]

    # Log model parameters and hyperparameters to mlflow
    mlflow_client.log_param(
        mlflow_run_id,
        "num_trainable_params",
        sum(p.numel() for p in networks.parameters() if p.requires_grad),
    )
    mlflow_client.log_param(
        mlflow_run_id,
        "num_non_trainable_params",
        sum(p.numel() for p in networks.parameters() if not p.requires_grad),
    )

    for iteration_idx in range(num_iterations):
        envs = [
            ParallelFiggieEnv(
                num_agents=5,
                num_steps=max_steps_per_game,
                env_id=env_id,
            )
            for env_id in range(num_games_per_iteration)
        ]

        vec_env = VectorizedParallelFiggieEnv(envs)
        game_runner = VectorizedGameRunner(
            vec_env,
            agent_pool,
            rng,
            num_steps=max_steps_per_game,
        )
        # collect data and retain full board snapshots for MAPPO joint critic
        snap_board: dict[
            tuple[int, Time],
            dict[AgentPoolID, EnvAgentStepSnapshot],
        ] = {}
        all_snaps_: list[
            tuple[int, Time, AgentPoolID, EnvAgentStepSnapshot]
        ] = []
        for time, snapshots in enumerate(
            game_runner.run(int(rng.integers(0, 9999999))),
        ):
            for env_idx, env_snapshots in enumerate(snapshots):
                snap_board[(env_idx, time)] = dict(env_snapshots)
                for agent_id, snapshot in env_snapshots.items():
                    all_snaps_.append((env_idx, time, agent_id, snapshot))
        # Get next step's beliefs for reward calculation
        snaps_indexed = {(e, t, a): snap for e, t, a, snap in all_snaps_}
        all_snaps: list[
            tuple[int, Time, AgentPoolID, EnvAgentStepSnapshot, BeliefType]
        ] = [
            (e, t, a, snap, snaps_indexed.get((e, t + 1, a), snap).belief)
            for e, t, a, snap in all_snaps_
        ]
        trainable_snaps = [
            snap
            for snap in all_snaps
            if agent_pool_agent_types[snap[2]] == "mappo"
        ]
        # Populate buffer StepRolloutBuffer < - snaps
        # Form batch TensorDict from buffer

        # Log beliefs
        for agent_type in ("card_counter", "mappo"):
            aggregates = aggregate_beliefs_by_suit_role(
                all_snaps,
                envs,
                agent_type,
                agent_pool_agent_types,
            )
            for suit_role, probs in aggregates.items():
                avg_probs = sum(probs) / len(probs) if probs else 0.0
                mlflow_client.log_metric(
                    mlflow_run_id,
                    f"belief/{agent_type}/avg_{suit_role}_prob",
                    avg_probs,
                    step=iteration_idx,
                )

        if not trainable_snaps:
            continue

        obs_list = [(snap[3].current_observation) for snap in trainable_snaps]
        obs = observations_to_tensordict_direct(obs_list)
        obs = preprocessors["nested"].preprocess(
            obs.unflatten_keys(),
            apply_soft_clip_on_prices=True,
            stage="inference",
        )
        obs = cast("TensorDict", obs["x"].to(get_device()))

        joint_x_pieces: list[TensorDict] = []
        for pool_idx in mappo_pool_indices:
            col_obs = [
                snap_board[(env_idx, time)][pool_idx].current_observation
                for env_idx, time, _pool_id, _snapshot, _nb in trainable_snaps
            ]
            joint_td = observations_to_tensordict_direct(col_obs)
            joint_td = preprocessors["nested"].preprocess(
                joint_td.unflatten_keys(),
                apply_soft_clip_on_prices=True,
                stage="inference",
            )
            joint_x_pieces.append(
                cast("TensorDict", joint_td["x"].to(get_device())),
            )
        joint_obs = stack(joint_x_pieces, dim=1)

        agent_slots = [
            mappo_pool_indices.index(pool_id)
            for _env_idx, _time, pool_id, _snapshot, _nb in trainable_snaps
        ]
        env_indices = [snap[0] for snap in trainable_snaps]
        time_indices = [snap[1] for snap in trainable_snaps]

        actions = [action_adaptor(snap[3].action) for snap in trainable_snaps]

        # Reward shaping:
        # Compute intermediate rewards by tracking state transitions.
        # For each PPO action at time t, we want to credit position changes that occur
        # between time t and t+1, using beliefs at t.

        adjusted_rewards = [
            compute_adjusted_reward_from_snapshot(snap[3], snap[4])
            for snap in trainable_snaps
        ]
        rewards = adjusted_rewards

        logprobs = [snap[3].extra["log_prob"] for snap in trainable_snaps]
        terminateds = [(snap[3].terminated) for snap in trainable_snaps]
        truncateds = [(snap[3].truncated) for snap in trainable_snaps]
        goal_codes = [
            SUIT_TO_CODE[envs[env_idx].goal_suit]
            for env_idx, *_ in trainable_snaps
        ]
        batch: TensorDict = TensorDict(
            {
                "observation": obs,
                "joint_observation": joint_obs,
                "action": torch.tensor(actions, device=get_device()),
                "reward": torch.tensor(rewards, device=get_device()),
                "logprob": torch.tensor(logprobs, device=get_device()),
                "termination": torch.tensor(
                    terminateds,
                    device=get_device(),
                ),
                "goal_suit_code": torch.tensor(goal_codes, device=get_device()),
                "truncation": torch.tensor(truncateds, device=get_device()),
                "agent_slot": torch.tensor(
                    agent_slots,
                    device=get_device(),
                    dtype=torch.long,
                ),
                "env_idx": torch.tensor(
                    env_indices,
                    device=get_device(),
                    dtype=torch.long,
                ),
                "time": torch.tensor(
                    time_indices,
                    device=get_device(),
                    dtype=torch.long,
                ),
            },
        )

        log_reward_metrics(
            mlflow_client=mlflow_client,
            mlflow_run_id=mlflow_run_id,
            iteration_idx=iteration_idx,
            snaps=trainable_snaps,
        )

        log_clairvoyant_metrics(
            snaps=all_snaps,
            envs=envs,
            agent_pool_agent_types=agent_pool_agent_types,
            mlflow_client=mlflow_client,
            mlflow_run_id=mlflow_run_id,
            iteration_idx=iteration_idx,
        )

        log_debug_metrics_for_iteration(
            mlflow_client=mlflow_client,
            mlflow_run_id=mlflow_run_id,
            iteration_idx=iteration_idx,
            num_iterations=num_iterations,
            snaps=all_snaps,
            agent_pool_agent_types=agent_pool_agent_types,
            envs=envs,
        )

        optimizer = lightning_module.configure_optimizers()
        metrics = {}
        for _epoch in range(num_epochs_per_iteration):
            loss, metrics = lightning_module.training_step(
                batch,
                iteration_idx,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]

        # Log all metrics to mlflow using the client
        global_step = iteration_idx  # * 100 + opt_step
        for metric_name, metric_value in metrics.items():
            mlflow_client.log_metric(
                mlflow_run_id,
                metric_name,
                metric_value,
                step=global_step,
            )

        mlflow_client.log_metric(
            mlflow_run_id,
            "actor_logstd_mean",
            float(
                networks.actor_logstd.detach().mean().cpu().numpy(),
            ),
            step=global_step,
        )


if __name__ == "__main__":
    description = input("Describe the run: ")
    train(
        max_steps_per_game=20,
        num_games_per_iteration=128,
        num_iterations=50,
        num_epochs_per_iteration=50,
        rng=np.random.default_rng(42),
        description=description,
    )
