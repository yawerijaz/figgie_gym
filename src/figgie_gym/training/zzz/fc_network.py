# pyright: reportPrivateImportUsage=false

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import lightning as pl
import pandas as pd
import structlog
import torch
from lightning.pytorch.loggers import MLFlowLogger

from figgie_gym.envs.common import SUITS
from figgie_gym.models import FCClassifier, FCDataModule
from figgie_gym.models.zzz.feature_description import (
    FeatureSpace,
    get_feature_list,
)
from figgie_gym.utilities import get_git_sha

logger = structlog.get_logger(__name__)


@dataclass
class FCTrainingConfig:
    # network
    hidden_dims: tuple[int, ...]
    dropout: float

    # data
    data_dir: Path
    feature_space: FeatureSpace
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
    mlflow_experiment: str = "Figgie Supervised (flat)"

    fast_dev_run: int = 300


example_config = FCTrainingConfig(
    hidden_dims=(64, 32),
    dropout=0.1,
    data_dir=Path("data/supervised_data_parsed_dev/"),
    feature_space="known_and_price",
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
    config: FCTrainingConfig,
) -> None:
    """Train FC Classifier."""
    logger.info("Loading data")
    dm = FCDataModule(
        data_dir=config.data_dir,
        feature_space=config.feature_space,
        data_split_pct=(
            config.train_pct,
            config.val_pct,
            1 - config.train_pct - config.val_pct,
        ),
        batch_size=config.batch_size,
        apply_soft_clip_on_prices=config.apply_soft_clip_on_prices,
        seed=config.seed,
    )

    model = FCClassifier(
        input_dim=len(get_feature_list(config.feature_space)),
        hidden_dims=list(config.hidden_dims),
        num_classes=4,
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
        "\n".join(get_feature_list(dm.feature_space)),
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

    full_df, full_dataloader = dm.complete_dataset_and_loader()
    logger.info("Begin predicting")

    logits = (
        torch.cat(
            cast(
                "list[torch.Tensor]",
                trainer.predict(model, dataloaders=full_dataloader),
            ),
            dim=0,
        )
        .detach()
        .cpu()
    )
    logger.info("Finished predicting")
    probs = torch.softmax(logits, dim=1).numpy()
    df = pd.concat(
        [
            full_df.reset_index(drop=True),
            pd.DataFrame(
                probs,
                columns=[f"prob_model_{sym}" for sym in SUITS],
            ).reset_index(drop=True),
        ],
        axis=1,
    )

    logger.info("Writing prediction results")
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "predictions.parquet"
        df.to_parquet(out_path, index=False)
        mlflow_logger.experiment.log_artifact(  # pyright: ignore[reportUnknownMemberType]
            mlflow_run_id,
            str(out_path),
            artifact_path="predictions",
        )
        logger.info("Logged predictions to MLflow", path=str(out_path))


if __name__ == "__main__":
    train(example_config)
