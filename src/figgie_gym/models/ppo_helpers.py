from figgie_gym.envs.common import (
    SUITS,
    ActionType,
    BeliefType,
    ObsType,
)
from figgie_gym.envs.vectorized_game_runner import (
    EnvAgentStepSnapshot,
)

FINAL_CARD_VALUE = 10.0


def total_suit_card_value_if_target(pos: float) -> float:
    """Compute portfolio value in a suit for reward shaping."""
    bonus = 100
    bonus_spread_over_cards = 6
    aggressive_factor = 1.1

    face_value = FINAL_CARD_VALUE * pos
    bonus_value = (
        bonus
        * (aggressive_factor**pos - 1)
        / (aggressive_factor**bonus_spread_over_cards - 1)
        if pos <= bonus_spread_over_cards
        else 0.0
    )
    # Empirically, adding bonus seems to hurt learning,
    # possibly by making the reward landscape too spiky.
    # Consider tuning down or removing bonus if learning is unstable.
    # Trying higher aggresiveness
    return face_value + bonus_value


def compute_total_porfolio_value_for_reward_shaping(
    obs: ObsType,
    belief: BeliefType,
) -> float:
    """Compute total portfolio value for reward shaping."""
    return (
        sum(
            total_suit_card_value_if_target(obs.per_suit[s].self_position)
            * belief[s]
            for s in SUITS
        )
        + obs.cash
    )


def compute_adjusted_reward_from_snapshot(
    snapshot: EnvAgentStepSnapshot,
    next_belief: BeliefType,
) -> float:
    """Compute reshaped reward."""
    curr_obs = snapshot.current_observation
    curr_belief = snapshot.belief
    next_obs = snapshot.next_observation
    terminated = snapshot.terminated
    truncated = snapshot.truncated

    current_portfolio_value = compute_total_porfolio_value_for_reward_shaping(
        curr_obs,
        curr_belief,
    )
    next_portfolio_value = (
        compute_total_porfolio_value_for_reward_shaping(
            next_obs,
            next_belief,
        )
        if not (terminated or truncated)
        else float(snapshot.reward)
    )

    max_portfolio_value = 200
    negative_cash_penalty_factor = 1
    return (
        next_portfolio_value
        - current_portfolio_value
        - negative_cash_penalty_factor * min(0, next_obs.cash + 5) ** 2
    ) / max_portfolio_value


def action_adaptor(
    action: ActionType,
) -> list[list[float]]:
    """Convert ActionType to a list of list of floats.

    Intended to be a preprocessing step of building a Tensor.
    """
    return [
        [
            float(action[s].quote_bid),
            float(action[s].quote_ask),
            float(action[s].snipe_bid),
            float(action[s].snipe_ask),
        ]
        for s in SUITS
    ]
