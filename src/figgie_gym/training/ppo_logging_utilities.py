from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from figgie_gym.envs.common import (
    SUIT_PARTNER_MAP,
    SUITS,
    BeliefType,
    ObsType,
    Time,
)
from figgie_gym.envs.parallel_figgie_env import ParallelFiggieEnv
from figgie_gym.envs.vectorized_game_runner import (
    AgentPoolID,
    EnvAgentStepSnapshot,
)
from figgie_gym.models.ppo_helpers import (
    FINAL_CARD_VALUE,
    compute_adjusted_reward_from_snapshot,
)

if TYPE_CHECKING:
    import mlflow


def extract_belief_array(
    snaps: list[
        tuple[int, Time, AgentPoolID, EnvAgentStepSnapshot, BeliefType]
    ],
) -> np.ndarray:
    arrs = [[float(snap[3].belief[s]) for s in SUITS] for snap in snaps]
    return np.array(arrs) if arrs else np.zeros((0, len(SUITS)))


def aggregate_beliefs_by_suit_role(
    snaps: list[
        tuple[int, Time, AgentPoolID, EnvAgentStepSnapshot, BeliefType]
    ],
    envs: list[ParallelFiggieEnv],
    agent_type: str,
    agent_pool_agent_types: list[str],
) -> dict[str, list[float]]:
    aggregated_beliefs: dict[str, list[float]] = {
        "target": [],
        "companion": [],
        "distractor": [],
    }
    filtered_snaps = [
        snap for snap in snaps if agent_pool_agent_types[snap[2]] == agent_type
    ]
    belief_array = extract_belief_array(filtered_snaps)
    for idx, (env_idx, *_) in enumerate(filtered_snaps):
        probs = belief_array[idx]
        goal_suit = envs[env_idx].goal_suit
        partner = SUIT_PARTNER_MAP[goal_suit]
        for s_idx, s in enumerate(SUITS):
            p = float(probs[s_idx])
            if s == goal_suit:
                aggregated_beliefs["target"].append(p)
            elif s == partner:
                aggregated_beliefs["companion"].append(p)
            else:
                aggregated_beliefs["distractor"].append(p)
    return aggregated_beliefs


def log_reward_metrics(
    mlflow_client: "mlflow.MlflowClient",
    mlflow_run_id: str,
    iteration_idx: int,
    snaps: list[
        tuple[int, Time, AgentPoolID, EnvAgentStepSnapshot, BeliefType]
    ],
) -> None:
    gamma = 0.99
    env_episode_rewards: dict[int, list[tuple[int, float]]] = {}
    undiscounted_rewards: list[float] = []
    discounted_rewards: list[float] = []
    for env_idx, time, _agent_id, snapshot, _next_belief in snaps:
        reward_value = float(snapshot.reward)
        undiscounted_rewards.append(reward_value)
        discounted_rewards.append((gamma**time) * reward_value)
        env_episode_rewards.setdefault(env_idx, []).append(
            (time, reward_value),
        )

    episode_returns = [
        sum(r for _, r in rewards_by_env)
        for rewards_by_env in env_episode_rewards.values()
    ]
    episode_discounted_returns = [
        sum((gamma**time) * r for time, r in rewards_by_env)
        for rewards_by_env in env_episode_rewards.values()
    ]
    avg_step_reward = (
        sum(undiscounted_rewards) / len(undiscounted_rewards)
        if undiscounted_rewards
        else 0.0
    )
    avg_step_discounted_reward = (
        sum(discounted_rewards) / len(discounted_rewards)
        if discounted_rewards
        else 0.0
    )
    avg_episode_reward = (
        sum(episode_returns) / len(episode_returns) if episode_returns else 0.0
    )
    avg_episode_discounted_reward = (
        sum(episode_discounted_returns) / len(episode_discounted_returns)
        if episode_discounted_returns
        else 0.0
    )

    mlflow_client.log_metric(
        mlflow_run_id,
        "reward/step/avg_undiscounted",
        avg_step_reward,
        step=iteration_idx,
    )
    mlflow_client.log_metric(
        mlflow_run_id,
        "reward/step/avg_discounted",
        avg_step_discounted_reward,
        step=iteration_idx,
    )
    mlflow_client.log_metric(
        mlflow_run_id,
        "reward/episode/avg_undiscounted",
        avg_episode_reward,
        step=iteration_idx,
    )
    mlflow_client.log_metric(
        mlflow_run_id,
        "reward/episode/avg_discounted",
        avg_episode_discounted_reward,
        step=iteration_idx,
    )


def log_debug_metrics_for_iteration(  # noqa: C901, PLR0913
    mlflow_client: "mlflow.MlflowClient",
    mlflow_run_id: str,
    iteration_idx: int,
    num_iterations: int,
    snaps: list[
        tuple[int, Time, AgentPoolID, EnvAgentStepSnapshot, BeliefType]
    ],
    agent_pool_agent_types: list[str],
    envs: list[ParallelFiggieEnv],
    should_log_every_n_iterations: int = 5,
) -> None:
    """Log per-iteration debug metrics for prices and positions by agent type and suit role.

    Aggregates metrics across all timesteps in the iteration and logs at reduced
    frequency (every 20 iterations + final iteration).
    """
    # Log iteration-level metrics at reduced frequency
    should_log = (
        (iteration_idx % should_log_every_n_iterations == 0)
        or (iteration_idx == 1)
        or (iteration_idx == num_iterations - 1)
    )
    if not should_log:
        return

    goal_suits = [env.goal_suit for env in envs]

    cash_by_agent: dict[tuple[Time, str], list[float]] = {}  # time, agent type

    # (time, agent_type) -> list of adjusted rewards for that timestep
    reward_by_agent_time: dict[tuple[Time, str], list[float]] = defaultdict(
        list,
    )

    # Collect prices and positions aggregated by (time, agent_type, suit_role, price_type/position).

    # [(time, agent_type, suit_role, suit_info_type) -> prices]
    suit_info_by_agent_and_role: dict[
        tuple[Time, str, str, str],
        list[float],
    ] = defaultdict(list)

    for env_idx, time, agent_id, snapshot, next_belief in snaps:
        agent_type = agent_pool_agent_types[agent_id]
        action = snapshot.action
        belief = snapshot.belief
        obs = snapshot.current_observation

        cash = obs.cash
        cash_by_agent.setdefault((time, agent_type), []).append(float(cash))

        reward_by_agent_time.setdefault((time, agent_type), []).append(
            compute_adjusted_reward_from_snapshot(snapshot, next_belief),
        )

        suit_role_mapping = dict.fromkeys(SUITS, "distractor")
        suit_role_mapping[goal_suits[env_idx]] = "target"
        suit_role_mapping[SUIT_PARTNER_MAP[goal_suits[env_idx]]] = "companion"

        for suit in SUITS:
            suit_info = {
                "quote_bid": action[suit].quote_bid,
                "quote_ask": action[suit].quote_ask,
                "quote_spread": (
                    action[suit].quote_ask - action[suit].quote_bid
                ),
                "snipe_bid": action[suit].snipe_bid,
                "snipe_ask": action[suit].snipe_ask,
                "position": obs.per_suit[suit].self_position,
                "belief_prob": belief[suit],
            }

            quote_fields = [
                suit_info["quote_bid"],
                suit_info["quote_ask"],
                suit_info["snipe_bid"],
                suit_info["snipe_ask"],
            ]
            suit_info["mixed_mid"] = sum(quote_fields) / len(quote_fields)

            if pricing := (snapshot.extra and snapshot.extra.get("pricing")):
                suit_info.update(
                    {f"pricing_{k}": v for k, v in pricing[suit].items()},
                )

            suit_role = suit_role_mapping[suit]

            for suit_info_type, suit_info_data in suit_info.items():
                suit_info_by_agent_and_role[
                    (time, agent_type, suit_role, suit_info_type)
                ].append(suit_info_data)

    for (time, agent_type), cash_list in cash_by_agent.items():
        if cash_list:
            mlflow_client.log_metric(
                mlflow_run_id,
                f"iterations/{iteration_idx}/{agent_type}/cash_avg",
                sum(cash_list) / len(cash_list),
                step=time,
            )

    # Log per-time average of adjusted (shaped) rewards per agent type
    for (time, agent_type), rewards_list in reward_by_agent_time.items():
        if rewards_list:
            mlflow_client.log_metric(
                mlflow_run_id,
                f"iterations/{iteration_idx}/{agent_type}/reward_avg",
                sum(rewards_list) / len(rewards_list),
                step=time,
            )

    # Log per-suit averages
    for (
        time,
        agent_type,
        suit_role,
        suit_info_type,
    ), vals in suit_info_by_agent_and_role.items():
        mlflow_client.log_metric(
            mlflow_run_id,
            f"iterations/{iteration_idx}/{agent_type}/{suit_role}/{suit_info_type}_avg",
            sum(vals) / len(vals) if vals else float("nan"),
            step=time,
        )


def log_clairvoyant_metrics(  # noqa: PLR0913
    snaps: list[tuple[int, int, int, EnvAgentStepSnapshot, BeliefType]],
    envs: list[ParallelFiggieEnv],
    agent_pool_agent_types: list[str],
    mlflow_client: "mlflow.MlflowClient",
    mlflow_run_id: str,
    iteration_idx: int,
) -> None:
    # Clairvoyant end-of-iteration PnL logging: replicate env final payout
    # using knowledge of the goal suit (clairvoyant). This matches the
    # scoring implemented in ParallelFiggieEnv.step where final rewards
    # are: q * FINAL_CARD_VALUE + (shared bonus if highest target holdings)
    # + cash. We compute final observation per (env, agent) and aggregate
    # per agent type across envs.
    clairvoyant_pnls_by_agent_type: dict[str, list[float]] = defaultdict(list)
    clairvoyant_bonuses_by_agent_type: dict[str, list[float]] = defaultdict(
        list,
    )

    # pick final observation per (env_idx, agent_id) using max time
    final_by_env: dict[int, dict[int, ObsType]] = defaultdict(dict)
    final_time: dict[tuple[int, int], int] = {}
    for env_idx, time, agent_id, snapshot, _next_belief in snaps:
        key = (env_idx, agent_id)
        prev_t = final_time.get(key)
        if (prev_t is None) or (time >= prev_t):
            final_time[key] = time
            final_by_env[env_idx][agent_id] = snapshot.current_observation

    for env_idx, ag_obs_map in final_by_env.items():
        if not ag_obs_map:
            continue
        goal_suit = envs[env_idx].goal_suit
        # goal suit holdings per agent in this env
        goal_holdings: dict[int, float] = {
            ag: float(obs.per_suit[goal_suit].self_position)
            for ag, obs in ag_obs_map.items()
        }

        goal_max = max(goal_holdings.values())
        total_bonus = 200 - FINAL_CARD_VALUE * sum(goal_holdings.values())
        awardees = [ag for ag, q in goal_holdings.items() if q == goal_max]
        bonus_per_awardee = total_bonus / len(awardees) if awardees else 0.0

        for ag, obs in ag_obs_map.items():
            q = goal_holdings[ag]
            cash = float(obs.cash)
            bonus = bonus_per_awardee if ag in awardees else 0.0
            pnl = q * FINAL_CARD_VALUE + bonus + cash
            agent_type = agent_pool_agent_types[ag]
            clairvoyant_pnls_by_agent_type.setdefault(agent_type, []).append(
                pnl,
            )
            clairvoyant_bonuses_by_agent_type.setdefault(agent_type, []).append(
                bonus,
            )

    for agent_type, pnls in clairvoyant_pnls_by_agent_type.items():
        mlflow_client.log_metric(
            mlflow_run_id,
            f"clairvoyant/{agent_type}/pnl_avg",
            sum(pnls) / len(pnls) if pnls else float("nan"),
            step=iteration_idx,
        )
    for agent_type, bonuses in clairvoyant_bonuses_by_agent_type.items():
        mlflow_client.log_metric(
            mlflow_run_id,
            f"clairvoyant/{agent_type}/bonus_avg",
            sum(bonuses) / len(bonuses) if bonuses else float("nan"),
            step=iteration_idx,
        )
