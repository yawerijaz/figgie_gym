import os
from dataclasses import dataclass
from pathlib import Path

import lightning as pl
import structlog
from lightning.pytorch.loggers import MLFlowLogger

from figgie_gym.models.suit_agent_equiv_classifier import (
    SuitAgentEquivClassifier,
)
from figgie_gym.pipelines.tensordict_dataloader import TensorDictDataModule
from figgie_gym.utilities import get_git_sha

logger = structlog.get_logger(__name__)


@dataclass
class TrainingConfig:
    # network
    hidden_dims: tuple[int, ...]
    dropout: float

    # data
    data_dir: Path
    train_pct: float
    val_pct: float
    apply_soft_clip_on_prices: bool

    # training
    epochs: int
    learning_rate: float
    batch_size: int
    seed: int
    dev: bool

    # logging
    description: str
    mlflow_uri: str = os.getenv("MLFLOW_TRACKING_URI") or ""
    mlflow_experiment: str = "Figgie Supervised (suit agent equiv)"

    fast_dev_run: int = 300


example_config = TrainingConfig(
    hidden_dims=(64, 32),
    dropout=0.1,
    data_dir=Path("data/supervised_data_parsed_dev/"),
    train_pct=0.8,
    val_pct=0.1,
    apply_soft_clip_on_prices=True,
    epochs=10,
    learning_rate=0.1,
    batch_size=1024,
    seed=42,
    dev=False,
    description="example",
)


def train(
    config: TrainingConfig,
) -> None:
    """Train FC Classifier."""
    logger.info("Loading data")
    dm = TensorDictDataModule(
        data_dir=config.data_dir,
        data_split_pct=(
            config.train_pct,
            config.val_pct,
            1 - config.train_pct - config.val_pct,
        ),
        batch_size=config.batch_size,
        apply_soft_clip_on_prices=config.apply_soft_clip_on_prices,
        seed=config.seed,
    )

    # 'step', 'steps_remaining', 'remaining_game_portion', 'cash',
    # per_suit: 'known_count', 'bid_price', 'bid_quantity', 'ask_price', 'ask_quantity', 'last_price', 'volume', 'self_position'
    # per_suit - agent_trade_summaries: 'buy_quantity', 'buy_consideration', 'sell_quantity', 'sell_consideration', 'min_net_quantity_change'

    top_level_num_feats = 4
    per_suit_num_feats = 8
    per_suit_agent_num_feat = 5
    per_suit_agent_num_out = 2

    main_body_in_dim = top_level_num_feats + 4 * (
        per_suit_num_feats + 5 * per_suit_agent_num_out
    )

    model = SuitAgentEquivClassifier(
        main_body_dims=(main_body_in_dim, config.hidden_dims, 4),
        per_suit_per_agent_dims=(
            per_suit_agent_num_feat,
            (),
            per_suit_agent_num_out,
        ),
        lr=config.learning_rate,
        dropout=config.dropout,
    )

    # Setup MLflow logger and checkpointing
    mlflow_logger = MLFlowLogger(
        experiment_name=config.mlflow_experiment,
        tracking_uri=config.mlflow_uri,
        tags={
            "mlflow.note.content": config.description,
            "git_commit_sha": get_git_sha(),
        },
        log_model="all",
    )

    mlflow_logger.experiment  # initialize property # pyright: ignore[reportUnknownMemberType]  # noqa: B018
    mlflow_run_id = mlflow_logger._run_id  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
    mlflow_logger.experiment.log_text(  # pyright: ignore[reportUnknownMemberType]
        mlflow_run_id,
        "everything",
        "feature_list",
    )
    logger.info("MLFlow run starts", mlflow_run_id=mlflow_run_id)

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        enable_progress_bar=True,
        logger=mlflow_logger,
        fast_dev_run=300 if config.dev else False,
    )

    logger.info("Begin training")
    trainer.fit(model, dm)
    logger.info("Finished training")


if __name__ == "__main__":
    train(example_config)
