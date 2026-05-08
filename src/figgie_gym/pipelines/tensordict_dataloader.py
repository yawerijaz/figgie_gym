# pyright: reportPrivateImportUsage=false
"""A Lightning Data Module to load tensordict data."""

from pathlib import Path

import lightning as pl
import structlog
import torch
from tensordict import TensorDict  # pyright: ignore[reportMissingTypeStubs]
from torch.utils.data import DataLoader, TensorDataset

from figgie_gym.pipelines.tendordict_preprocess import (
    PreProcessorType,
    TensorDictPreprocessor,
    preprocessors,
)
from figgie_gym.utilities import identity

logger = structlog.get_logger(__name__)


class TensorDictDataModule(pl.LightningDataModule):
    """Load game observations data.

    Generates train/val/test tensors of shape (N, input_dim).
    """

    datasets: dict[str, TensorDataset]

    def __init__(  # noqa: PLR0913
        self,
        data_dir: Path,
        data_split_pct: tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        apply_soft_clip_on_prices: bool = True,  # noqa: FBT001, FBT002
        seed: int = 42,
        preprocessor: PreProcessorType = "original",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.data_split_pct = data_split_pct
        self.batch_size = batch_size
        epsilon = 1e-4
        if abs(sum(data_split_pct) - 1) > epsilon:
            raise ValueError(data_split_pct)
        self.seed = seed
        self.apply_soft_clip_on_prices = apply_soft_clip_on_prices
        self.preprocessor: PreProcessorType = preprocessor

    def setup(self, stage: str = "train") -> None:
        logger.info("Loading tensordict", datadir=self.data_dir)
        tensordict = TensorDict.load_memmap(self.data_dir)
        logger.info("Preprocessing tensordict")
        preprocessor = preprocessors[self.preprocessor]
        tensordict = self.preprocessor_.preprocess(
            tensordict,
            self.apply_soft_clip_on_prices,
            stage,  # pyright: ignore[reportArgumentType]
        )

        g = torch.Generator()
        g.manual_seed(self.seed)
        self.datasets = preprocessor.split_dataset(
            tensordict,
            self.data_split_pct,
            g,
        )

    @property
    def preprocessor_(
        self,
    ) -> TensorDictPreprocessor:
        return preprocessors[self.preprocessor]

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, ...]]:
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=9,
            persistent_workers=True,
            collate_fn=identity,
        )

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, ...]]:
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            num_workers=9,
            persistent_workers=True,
            collate_fn=identity,
        )

    def test_dataloader(self) -> DataLoader[tuple[torch.Tensor, ...]]:
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=9,
            persistent_workers=True,
            collate_fn=identity,
        )
