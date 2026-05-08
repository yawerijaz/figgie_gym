import numpy as np

from figgie_gym.agent.common import AGENT_NAME_CODES
from figgie_gym.envs.common import (
    SUITS,
    ActionOnSuit,
    ActionType,
    Agent,
    BeliefType,
    ExtraType,
    ObsType,
)


class NoiseAgent(Agent):
    agent_type_code = AGENT_NAME_CODES["NoiseAgent"]

    def act(
        self,
        random_number_generator: np.random.Generator,
        observation: ObsType,
    ) -> tuple[BeliefType, ActionType, ExtraType]:
        suit_obs = observation.per_suit
        num_obs = len(suit_obs)
        quote_bid_ask = random_number_generator.uniform(0, 1, size=(num_obs, 2))
        quote_bid_ask.sort(axis=1)
        delta = random_number_generator.uniform(0, 1, size=(num_obs, 2)) * 0.8
        snipe_bid_ask = quote_bid_ask[:, [1, 0]] + delta * [-1, 1]
        scale = 10
        return (
            dict.fromkeys(SUITS, 1 / len(SUITS)),
            {
                sym: ActionOnSuit(
                    quote_bid=quote_bid_ask[i, 0] * scale,
                    quote_ask=quote_bid_ask[i, 1] * scale,
                    snipe_bid=snipe_bid_ask[i, 0] * scale,
                    snipe_ask=snipe_bid_ask[i, 1] * scale,
                )
                for i, sym in enumerate(suit_obs)
            },
            None,
        )

    def act_batch(
        self,
        random_number_generator: np.random.Generator,
        observations: list[ObsType],
    ) -> list[tuple[BeliefType, ActionType, None]]:
        num_obs = len(observations)
        num_suit = len(SUITS)
        quote_bid_ask = random_number_generator.uniform(
            0,
            1,
            size=(num_obs, num_suit, 2),
        )
        quote_bid_ask.sort(axis=2)
        delta = (
            random_number_generator.uniform(0, 1, size=(num_obs, num_suit, 2))
            * 0.8
        )
        snipe_bid_ask = quote_bid_ask[:, :, [1, 0]] + delta * [-1, 1]
        scale = 10
        return [
            (
                dict.fromkeys(SUITS, 1 / len(SUITS)),
                {
                    sym: ActionOnSuit(
                        quote_bid=quote_bid_ask[i, sym_num, 0] * scale,
                        quote_ask=quote_bid_ask[i, sym_num, 1] * scale,
                        snipe_bid=snipe_bid_ask[i, sym_num, 0] * scale,
                        snipe_ask=snipe_bid_ask[i, sym_num, 1] * scale,
                    )
                    for sym_num, sym in enumerate(SUITS)
                },
                None,
            )
            for i in range(num_obs)
        ]
