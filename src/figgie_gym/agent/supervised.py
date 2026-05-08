# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, Literal, override

import numpy as np
import torch
from numpy import float64, int32

from figgie_gym.envs.common import (
    SUITS,
    ActionOnSuit,
    ActionType,
    Agent,
    BeliefType,
    ExtraType,
    ObsType,
)
from figgie_gym.models.suit_agent_equiv_classifier import (
    SuitAgentEquivClassifier,
)
from figgie_gym.pipelines.observations_to_tensordict import (
    observations_to_tensordict_direct,
)
from figgie_gym.pipelines.tendordict_preprocess import (
    PreProcessorType,
    preprocessors,
)
from figgie_gym.utilities import get_device

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SimplifiedExpectedValueGeometricAggressive:
    MAX_HOLDING = 12

    def __init__(
        self,
        full_pot: int,
        goal_symbol_value: int,
        num_symbols: int,
        late_aggressiveness_factor: float,
    ) -> None:
        self.full_pot = full_pot
        self.goal_symbol_value = goal_symbol_value
        self.num_symbols = num_symbols
        self.late_aggressiveness_factor = max(
            late_aggressiveness_factor,
            1 + 1e-6,
        )

        self.symbol_value = {
            "intrinsic": self.compute_symbol_value_goal_only("intrinsic"),
            "pot": self.compute_symbol_value_goal_only("pot"),
            "all": self.compute_symbol_value_goal_only("all"),
        }

    def compute_symbol_value_goal_only(
        self,
        include: Literal["intrinsic", "pot", "all"] = "all",
    ) -> NDArray[float64]:
        r = self.late_aggressiveness_factor
        # For simplicity, we assume there are 10 goal cards, instead of modeling whether the goal is 8 or 10.
        goal_quantity = 10
        holding_guarantee_pot = goal_quantity // 2 + 1
        pot = self.full_pot - goal_quantity * self.goal_symbol_value
        value_of_first = pot * (1 - r) / (1 - r**holding_guarantee_pot)

        value_of_each = value_of_first * r ** np.arange(self.MAX_HOLDING + 1)
        mask = np.arange(self.MAX_HOLDING + 1) < holding_guarantee_pot
        value_of_each = np.where(mask, value_of_each, 0).round()

        match include:
            case "all":
                return value_of_each + self.goal_symbol_value
            case "intrinsic":
                return np.full(
                    self.MAX_HOLDING + 1,
                    float(self.goal_symbol_value),
                    dtype=float64,
                )
            case "pot":
                return value_of_each

    def compute_expected_value(
        self,
        goal_probs: NDArray[float64],
        current_holdings: NDArray[int32],
        include: Literal["intrinsic", "pot", "all"] = "all",
    ) -> NDArray[float64]:
        val_is_goal = self.symbol_value[include]
        # no value if not goal
        indices = np.maximum(
            0,
            np.minimum(current_holdings, self.MAX_HOLDING),
        )
        return goal_probs * val_is_goal[indices]

    def breakeven_buy_price(
        self,
        goal_probs: NDArray[float64],
        current_holdings: NDArray[int32],
    ) -> NDArray[float64]:
        return self.compute_expected_value(goal_probs, current_holdings)

    def breakeven_sell_price(
        self,
        goal_probs: NDArray[float64],
        current_holdings: NDArray[int32],
    ) -> NDArray[float64]:
        return self.compute_expected_value(
            goal_probs,
            current_holdings - 1,
        )


class SupervisedModelAgent(Agent):
    def __init__(
        self,
        model: torch.nn.Module,
        ev_calculator: SimplifiedExpectedValueGeometricAggressive,
        quote_spread: int,
        apply_soft_clip: bool,  # noqa: FBT001
        preprocessor: PreProcessorType,
    ) -> None:
        self.model = model.to(get_device())
        self.ev_calculator = ev_calculator
        self.quote_spread = quote_spread
        self.apply_soft_clip = apply_soft_clip
        self.preprocessor: PreProcessorType = preprocessor
        self.model.eval()

    @override
    def params(self) -> dict[str, Any]:
        return super().params() | {
            "quote_spread": self.quote_spread,
            "late_aggressiveness_factor": self.ev_calculator.late_aggressiveness_factor,
            "preprocessor": self.preprocessor,
            "model_class": self.model.__class__.__name__,
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        ev_calculator: SimplifiedExpectedValueGeometricAggressive,
        quote_spread: int,
        apply_soft_clip: bool,  # noqa: FBT001
        preprocessor: PreProcessorType,
    ) -> SupervisedModelAgent:
        # Fix for WeightsUnpickler error when loading checkpoints containing pathlib objects
        with torch.serialization.safe_globals(
            [pathlib.PosixPath, pathlib.Path, pathlib.PurePosixPath],
        ):
            model = SuitAgentEquivClassifier.load_from_checkpoint(  # pyright: ignore[reportUnknownMemberType]
                checkpoint_path,
            )
        return cls(
            model,
            ev_calculator,
            quote_spread,
            apply_soft_clip,
            preprocessor,
        )

    def act(
        self,
        random_number_generator: np.random.Generator,  # noqa: ARG002
        observation: ObsType,
    ) -> tuple[BeliefType, ActionType, ExtraType]:
        probs = self.predict_probs([observation])[0]
        return (
            dict(
                zip(SUITS, probs, strict=True),
            ),
            self.compute_action_from_probs(probs, observation),
            None,
        )

    def act_batch(
        self,
        random_number_generator: np.random.Generator,  # noqa: ARG002
        observations: list[ObsType],
    ) -> list[tuple[BeliefType, ActionType, ExtraType]]:
        probs_batch = self.predict_probs(observations)
        beliefs: list[BeliefType] = [
            dict(zip(SUITS, probs, strict=True)) for probs in probs_batch
        ]
        actions: list[ActionType] = [
            self.compute_action_from_probs(probs, obs)
            for probs, obs in zip(probs_batch, observations, strict=True)
        ]
        return list(zip(beliefs, actions, [None] * len(beliefs), strict=True))

    def predict_probs(self, observations: list[ObsType]) -> NDArray[float64]:
        if not observations:
            return np.empty((0, 4), dtype=float64)

        td = observations_to_tensordict_direct(observations)

        preprocessed = preprocessors[self.preprocessor].preprocess(
            td.unflatten_keys(),
            self.apply_soft_clip,
            "inference",
        )
        model_input = preprocessed["x"].to(device=get_device())  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

        with torch.no_grad():
            logits = self.model(model_input)  # pyright: ignore[reportUnknownMemberType]
            return torch.softmax(logits, dim=1).cpu().numpy()

    def compute_action_from_probs(
        self,
        probs: NDArray[float64],
        observation: ObsType,
    ) -> ActionType:
        suit_obs = observation.per_suit
        my_holdings = np.array(
            [int(suit_obs[s].self_position) for s in SUITS],
            dtype=int32,
        )

        fair_buy = self.ev_calculator.breakeven_buy_price(probs, my_holdings)
        fair_sell = self.ev_calculator.breakeven_sell_price(probs, my_holdings)

        fair_mid = (fair_buy + fair_sell) / 2
        quote_bid = (
            np.minimum(fair_mid, fair_buy) - self.quote_spread / 2
        ).clip(min=0)
        quote_ask = np.maximum(fair_mid, fair_sell) + self.quote_spread / 2

        return {
            sym: ActionOnSuit(
                quote_bid=float(quote_bid[j]),
                quote_ask=float(quote_ask[j]),
                snipe_bid=float(fair_buy[j]),
                snipe_ask=float(fair_sell[j]),
            )
            for j, sym in enumerate(SUITS)
        }
