"""Run model training script.

Example:
python scripts/train.py --epochs 30 net-config:flat-net-config --net-config.preprocessor raw_flat
python scripts/train.py --epochs 30 net-config:equivariant-net-config

"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import structlog
import tyro
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger

from figgie_gym.models.equivariant import EquivariantClassifier
from figgie_gym.models.fc_tensordict_classifier import FCTensorDictClassifier
from figgie_gym.models.suit_agent_equiv_classifier import (
    SuitAgentEquivClassifier,
)
from figgie_gym.pipelines.tensordict_dataloader import TensorDictDataModule
from figgie_gym.pipelines.tensordict_flatten_features import get_feature_list
from figgie_gym.utilities import get_git_sha

if TYPE_CHECKING:
    from mlflow import MlflowClient

logger = structlog.get_logger(__name__)


@dataclass
class FlatNetConfig:
    preprocessor: Literal[
        "known_count_flat",
        "known_and_price_flat",
        "raw_flat",
    ] = "raw_flat"


@dataclass
class EquivariantNetConfig:
    """Config for the EquivariantClassifier model."""

    trade_summary_hidden_dims: tuple[int, ...] = (32,)
    trade_summary_embed_dim: int = 16
    known_trade_aggregate_hidden_dims: tuple[int, ...] = (32,)
    known_trade_aggregate_embed_dim: int = 12
    suit_embed_hidden_dims: tuple[int, ...] = (32,)
    suit_embed_dim: int = 16
    other_suit_context_hidden_dims: tuple[int, ...] = (32,)
    belief_hidden_dims: tuple[int, ...] = (32,)
    preprocessor: str = "nested"


@dataclass
class SuitAgentEquivariantNetConfig:
    preprocessor = "nested"


type NetConfig = (
    FlatNetConfig | EquivariantNetConfig | SuitAgentEquivariantNetConfig
)


@dataclass
class TrainingConfig:
    # network
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.1

    net_config: NetConfig = field(default_factory=EquivariantNetConfig)

    # data
    data_dir: Path = Path("data/multi/tensordict/")
    data_dir_dev: Path = Path("data/multi_dev/tensordict/")
    train_pct: float = 0.8
    val_pct: float = 0.1
    apply_soft_clip_on_prices: bool = True

    # training
    epochs: int = 5
    learning_rate: float = 1e-3
    batch_size: int = 1024
    seed: int = 42
    dev: bool = False

    # logging
    description: str = ""
    mlflow_uri: str = os.getenv("MLFLOW_TRACKING_URI") or ""
    mlflow_experiment: str = "Figgie Supervised (same network size test)"

    fast_dev_run: int = 300


if __name__ == "__main__":
    args = tyro.cli(TrainingConfig)
    data_module = TensorDictDataModule(
        data_dir=args.data_dir,
        data_split_pct=(
            args.train_pct,
            args.val_pct,
            1 - args.train_pct - args.val_pct,
        ),
        batch_size=args.batch_size,
        apply_soft_clip_on_prices=args.apply_soft_clip_on_prices,
        seed=args.seed,
        preprocessor=args.net_config.preprocessor,  # pyright: ignore[reportArgumentType]
    )

    match args.net_config:
        case FlatNetConfig(preprocessor=preprocessor):
            in_dim = len(get_feature_list(preprocessor))
            out_dim = 4
            model_module = FCTensorDictClassifier(
                main_body_dims=(in_dim, args.hidden_dims, out_dim),
                dropout=args.dropout,
                lr=args.learning_rate,
            )
        case EquivariantNetConfig(
            trade_summary_hidden_dims=ts_hidden,
            trade_summary_embed_dim=ts_embed,
            known_trade_aggregate_hidden_dims=kta_hidden,
            known_trade_aggregate_embed_dim=kta_embed,
            suit_embed_hidden_dims=se_hidden,
            suit_embed_dim=se_dim,
            other_suit_context_hidden_dims=osc_hidden,
            belief_hidden_dims=belief_hidden,
        ):
            model_module = EquivariantClassifier(
                trade_summary_input_dim=5,
                trade_summary_hidden_dims=ts_hidden,
                trade_summary_embed_dim=ts_embed,
                known_trade_aggregate_hidden_dims=kta_hidden,
                known_trade_aggregate_embed_dim=kta_embed,
                private_holding_in_dim=8,
                suit_embed_hidden_dims=se_hidden,
                suit_embed_dim=se_dim,
                other_suit_context_hidden_dims=osc_hidden,
                belief_hidden_dims=belief_hidden,
                lr=args.learning_rate,
            )
        case SuitAgentEquivariantNetConfig():
            top_level_num_feats = 4
            per_suit_num_feats = 8
            per_suit_agent_num_feat = 5
            per_suit_agent_num_out = 2

            main_body_in_dim = top_level_num_feats + 4 * (
                per_suit_num_feats + 5 * per_suit_agent_num_out
            )

            model_module = SuitAgentEquivClassifier(
                main_body_dims=(main_body_in_dim, args.hidden_dims, 4),
                per_suit_per_agent_dims=(
                    per_suit_agent_num_feat,
                    (),
                    per_suit_agent_num_out,
                ),
                lr=args.learning_rate,
                dropout=args.dropout,
            )

    description = (
        args.description
        or ("dev" if args.dev else "")
        or input("Describe the run: ")
    )

    artifact_dir = Path("mlartifacts/")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    mlflow_logger = MLFlowLogger(
        experiment_name=args.mlflow_experiment,
        tracking_uri=args.mlflow_uri,
        tags={
            "mlflow.note.content": description,
            "git_commit_sha": get_git_sha(),
        },
        log_model="all",
        artifact_location=str(artifact_dir.resolve()),
    )

    mlflow_client = cast(
        "MlflowClient",
        mlflow_logger.experiment,  # initialize property # pyright: ignore[reportUnknownMemberType]
    )
    mlflow_run_id = cast("str", mlflow_logger._run_id)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
    mlflow_client.log_text(  # pyright: ignore[reportUnknownMemberType]
        mlflow_run_id,
        "everything",
        "feature_list",
    )
    mlflow_client.log_param(
        mlflow_run_id,
        "num_trainable_params",
        sum(p.numel() for p in model_module.parameters() if p.requires_grad),
    )
    mlflow_client.log_param(
        mlflow_run_id,
        "num_non_trainable_params",
        sum(
            p.numel() for p in model_module.parameters() if not p.requires_grad
        ),
    )
    logger.info("MLFlow run starts", mlflow_run_id=mlflow_run_id)

    trainer = Trainer(
        max_epochs=args.epochs,
        enable_progress_bar=True,
        logger=mlflow_logger,
        default_root_dir=str(artifact_dir),
        fast_dev_run=300 if args.dev else False,
    )

    logger.info("Begin training")
    trainer.fit(model_module, data_module)
    logger.info("Finished training")
