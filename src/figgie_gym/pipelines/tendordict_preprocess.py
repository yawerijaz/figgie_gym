# pyright: reportPrivateImportUsage=false
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import ClassVar, Literal, cast

import structlog
import torch
from tensordict import (  # pyright: ignore[reportMissingTypeStubs]
    TensorDict,  # pyright: ignore[reportMissingTypeStubs]
    stack,  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
)
from torch.utils.data import TensorDataset

from figgie_gym.envs.common import SUITS
from figgie_gym.models.benchmark import card_counter_prediction
from figgie_gym.pipelines.tensordict_flatten_features import get_feature_list
from figgie_gym.utilities import identity, soft_clip

logger = structlog.get_logger(__name__)

DEFAULT_PRICE = 5
DEFAULT_QUANTITY = 0


def imputed_value_given_key(key: str) -> float | None:
    if key.endswith(("last_price", "ask.price", "bid.price")):
        return DEFAULT_PRICE
    if key.endswith(("ask.quantity", "bid.quantity")):
        return DEFAULT_QUANTITY
    return None


def impute_nan(
    imputed_value_given_key: Callable[[str], float | None],
    key: str,
    tensor: torch.Tensor,
) -> torch.Tensor:
    if (val := imputed_value_given_key(key)) is not None:
        return tensor.nan_to_num(val)

    return tensor


class TensorDictPreprocessor(ABC):
    @abstractmethod
    def preprocess(
        self,
        tensordict: TensorDict,
        apply_soft_clip_on_prices: bool,  # noqa: FBT001
        stage: Literal["train", "inference"] = "train",
    ) -> TensorDict: ...

    def calculate_common_fields(self, tensordict: TensorDict) -> TensorDict:
        known_counts = stack(
            [
                cast(
                    "torch.Tensor",
                    tensordict.unflatten_keys()[
                        "current_observations",
                        "per_suit",
                        sym,
                        "known_count",
                    ],
                )
                for sym in SUITS
            ],
            dim=1,
        )
        y_card_counter = torch.Tensor(
            card_counter_prediction(known_counts.numpy()),
        ).clip(min=1e-6, max=1 - 1e-6)
        y = cast("torch.Tensor", tensordict["game_info.goal_suit_code"])
        game = cast("torch.Tensor", tensordict["env_id"])
        step = cast("torch.Tensor", tensordict["game_runner_step"])
        return TensorDict(
            {
                "y": y,
                "y_card_counter": y_card_counter,
                "game": game,
                "step": step,
            },
            batch_size=len(tensordict),
        )

    def split_dataset(
        self,
        processed_dict: TensorDict,
        data_split_pct: tuple[float, float, float],
        torch_rng: torch.Generator,
    ) -> dict[str, TensorDataset]:
        total_experiments = int(processed_dict["game"].max()) + 1  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        game_index = torch.randperm(total_experiments, generator=torch_rng)
        train_size = int(total_experiments * data_split_pct[0])
        val_size = int(total_experiments * data_split_pct[1])
        logger.info(
            "Splitting tensordict",
            total_experiments=total_experiments,
            train_size=train_size,
            val_size=val_size,
        )

        data_game_split = {
            "train": game_index[:train_size],
            "val": game_index[train_size : train_size + val_size],
            "test": game_index[train_size + val_size :],
        }

        data_mask = {
            name: torch.isin(
                cast("torch.Tensor", processed_dict["game"]),
                game_split,
            )
            for name, game_split in data_game_split.items()
        }

        return {
            name: cast("TensorDict", processed_dict[mask])
            for name, mask in data_mask.items()
        }


class OriginalTensorDictPreprocessor(TensorDictPreprocessor):
    target_field = "hidden.goal_suit_code"
    feature_fields: ClassVar[list[str]] = [
        "step",
        "steps_remaining",
        "remaining_game_portion",
        "cash",
        "per_suit",
    ]

    def split_dataset(
        self,
        processed_dict: TensorDict,
        data_split_pct: tuple[float, float, float],
        torch_rng: torch.Generator,
    ) -> dict[str, TensorDataset]:
        total_experiments = int(processed_dict["hidden.experiment"].max()) + 1  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        game_index = torch.randperm(total_experiments, generator=torch_rng)
        train_size = int(total_experiments * data_split_pct[0])
        val_size = int(total_experiments * data_split_pct[1])
        logger.info(
            "Splitting tensordict",
            total_experiments=total_experiments,
            train_size=train_size,
            val_size=val_size,
        )

        datasets = dict[str, TensorDataset]()
        for name, mask in [
            (
                "train",
                torch.isin(
                    cast("torch.Tensor", processed_dict["hidden.experiment"]),
                    game_index[:train_size],
                ),
            ),
            (
                "val",
                torch.isin(
                    cast("torch.Tensor", processed_dict["hidden.experiment"]),
                    game_index[train_size : train_size + val_size],
                ),
            ),
            (
                "test",
                torch.isin(
                    cast("torch.Tensor", processed_dict["hidden.experiment"]),
                    game_index[train_size + val_size :],
                ),
            ),
        ]:
            data = cast("TensorDict", processed_dict[mask])
            x = data.select(*self.feature_fields)
            y = cast("torch.Tensor", data[self.target_field])
            y_card_counter = torch.Tensor(
                card_counter_prediction(
                    data["per_suit", "known_count"].numpy(),  # pyright: ignore[reportUnknownMemberType,reportArgumentType]
                ),
            ).clip(min=1e-6, max=1 - 1e-6)

            datasets[name] = TensorDict(
                {
                    "x": x,
                    "y": y,
                    "y_card_counter": y_card_counter,
                    "game": data["hidden.experiment"],
                    "step": data["step"],
                },
                batch_size=len(data),
            )
            logger.info("Splitted tensordict", dataset=name, shape=data.shape)
        return datasets

    def preprocess(
        self,
        tensordict: TensorDict,
        apply_soft_clip_on_prices: bool,  # noqa: FBT001
        stage: Literal["train", "inference"] = "train",  # noqa: ARG002
    ) -> TensorDict:
        market_price_fields = [
            ("per_suit", "last_price"),
            ("per_suit", "bid_price"),
            ("per_suit", "ask_price"),
        ]

        for price_field in market_price_fields:
            tensordict[price_field] = tensordict[price_field].nan_to_num(  # pyright: ignore[reportUnknownMemberType]
                DEFAULT_PRICE,
            )
            if apply_soft_clip_on_prices:
                tensordict[price_field] = soft_clip(tensordict[price_field])  # pyright: ignore[reportUnknownArgumentType]

        market_quantity_fields = [
            ("per_suit", "bid_quantity"),
            ("per_suit", "ask_quantity"),
        ]
        for quantity_field in market_quantity_fields:
            tensordict[quantity_field] = tensordict[quantity_field].nan_to_num(  # pyright: ignore[reportUnknownMemberType]
                0,
            )

        return tensordict


class FlatTensorDictPreprocessor(TensorDictPreprocessor):
    target_field = "game_info.goal_suit_code"

    def __init__(self, feature_fields: list[str]) -> None:
        self.feature_fields = feature_fields

    def preprocess(
        self,
        tensordict: TensorDict,
        apply_soft_clip_on_prices: bool,  # noqa: FBT001
        stage: Literal["train", "inference"] = "train",
    ) -> TensorDict:
        x = cast(
            "TensorDict",
            tensordict.unflatten_keys()["current_observations"].flatten_keys(),  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        ).select(*self.feature_fields)
        for k, v in x.items():  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            x[k] = impute_nan(
                imputed_value_given_key,
                k,
                cast("torch.Tensor", v),
            )
            if apply_soft_clip_on_prices and "price" in k:
                x[k] = soft_clip(cast("torch.Tensor", x[k]))
        if stage == "inference":
            return TensorDict({"x": x})
        return self.calculate_common_fields(tensordict).set("x", x)  # pyright: ignore[reportUnknownMemberType]


class NestedTensorDictPreprocessor(TensorDictPreprocessor):
    target_field: ClassVar[list[str]] = ["game_info", "goal_suit_code"]

    def preprocess(
        self,
        tensordict: TensorDict,
        apply_soft_clip_on_prices: bool,  # noqa: FBT001
        stage: Literal["train", "inference"] = "train",
    ) -> TensorDict:
        price_transform_func = (
            soft_clip if apply_soft_clip_on_prices else identity
        )
        obs = cast(
            "TensorDict",
            tensordict.unflatten_keys()["current_observations"],
        )

        pur_suit, x = cast("Iterable[TensorDict]", obs.split_keys(["per_suit"]))  # pyright: ignore[reportUnknownMemberType]

        per_suit = stack(
            [cast("TensorDict", pur_suit["per_suit", sym]) for sym in SUITS],
            -1,
        )
        (
            market_quote,
            last_price,
            self_trade_summary,
            other_trade_summaries,
            per_suit,
        ) = cast(
            "Iterable[TensorDict]",
            per_suit.split_keys(  # pyright: ignore[reportUnknownMemberType]
                ["market_quote"],
                ["last_price"],
                ["self_trade_summary"],
                ["other_trade_summaries"],
            ),
        )
        market_quote_fields = {
            f"market_{way}_{field}": transform_func(
                cast(
                    "torch.Tensor",
                    market_quote["market_quote", way, field],
                ).nan_to_num(fill_nan_value),
            )
            for way in ["bid", "ask"]
            for (field, fill_nan_value, transform_func) in [
                ("price", 5, price_transform_func),
                ("quantity", 0, identity),
            ]
        }
        trade_summaries = stack(
            [
                cast("TensorDict", self_trade_summary["self_trade_summary"]),
                *cast(
                    "Iterable[TensorDict]",
                    cast(
                        "TensorDict",
                        other_trade_summaries["other_trade_summaries"],
                    ).values(),  # pyright: ignore[reportUnknownMemberType]
                ),
            ],
            -1,
        )
        x = x.set(  # pyright: ignore[reportUnknownMemberType]
            "per_suit",
            per_suit.set(  # pyright: ignore[reportUnknownMemberType]
                "last_price",
                (
                    price_transform_func(
                        cast(
                            "torch.Tensor",
                            last_price["last_price"],
                        ).nan_to_num(5),
                    )
                ),
            )
            .set("trade_summaries", trade_summaries)
            .update(market_quote_fields),
        )
        if stage == "inference":
            return TensorDict({"x": x})
        return self.calculate_common_fields(tensordict).set("x", x)  # pyright: ignore[reportUnknownMemberType]


type PreProcessorType = Literal[
    "known_count_flat",
    "known_and_price_flat",
    "raw_flat",
    "original",
    "nested",
]

preprocessors: dict[PreProcessorType, TensorDictPreprocessor] = {
    "original": OriginalTensorDictPreprocessor(),
    "known_count_flat": FlatTensorDictPreprocessor(
        get_feature_list("known_count_flat"),
    ),
    "known_and_price_flat": FlatTensorDictPreprocessor(
        get_feature_list("known_and_price_flat"),
    ),
    "raw_flat": FlatTensorDictPreprocessor(get_feature_list("raw_flat")),
    "nested": NestedTensorDictPreprocessor(),
}
