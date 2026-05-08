# pyright: reportPrivateImportUsage=false

from itertools import pairwise
from typing import Literal, cast

import lightning as pl
import structlog
import torch
from tensordict import (  # pyright: ignore[reportMissingTypeStubs]
    TensorDict,  # pyright: ignore[reportUnknownVariableType]
)
from torch import Tensor, nn
from torch.optim.adam import Adam

logger = structlog.get_logger(__name__)


class SequentialLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (),
        output_dim: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        dims = [input_dim, *hidden_dims, output_dim]

        layers = list[nn.Module]()

        io_dims = list(pairwise(dims))
        for io_dim_num, (i_dim, o_dim) in enumerate(io_dims):
            layers.append(nn.LayerNorm(i_dim))
            layers.append(nn.Linear(i_dim, o_dim))
            if io_dim_num != len(io_dims) - 1:
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


type LossModule = nn.Module


class CommonTensorDictLightningModule(pl.LightningModule):
    loss_fn: LossModule
    lr: float | Tensor

    def training_step(
        self,
        batch: TensorDict,
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        x, y, y_card_counter = unpack(batch)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True)

        benchmark_loss = self.loss_fn(torch.logit(y_card_counter), y)
        benchmark_acc = (
            (torch.argmax(y_card_counter, dim=1) == y).float().mean()
        )
        self.log(
            "train/benchmark/loss",
            benchmark_loss,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/benchmark/acc",
            benchmark_acc,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(
        self,
        batch: TensorDict,
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        x, y, y_card_counter = unpack(batch)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        benchmark_loss = self.loss_fn(torch.logit(y_card_counter), y)
        benchmark_acc = (
            (torch.argmax(y_card_counter, dim=1) == y).float().mean()
        )
        self.log(
            "val/benchmark/loss",
            benchmark_loss,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/benchmark/acc",
            benchmark_acc,
            on_step=False,
            on_epoch=True,
        )

    def predict_dataloader(self) -> torch.Tensor:
        raise NotImplementedError

    def configure_optimizers(self) -> Adam:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def parameter_count(
        self,
        parameter_type: Literal["trainable", "non_trainable"],
    ) -> int:
        match parameter_type:
            case "trainable":
                return sum(
                    p.numel() for p in self.parameters() if p.requires_grad
                )
            case "non_trainable":
                return sum(
                    p.numel() for p in self.parameters() if not p.requires_grad
                )


def unpack(batch: TensorDict) -> tuple[TensorDict, Tensor, Tensor]:
    return (
        cast("TensorDict", batch["x"].to(dtype=torch.float32)),  # pyright: ignore[reportUnknownMemberType, reportCallIssue, reportArgumentType]
        cast("Tensor", batch["y"]),
        cast("Tensor", batch["y_card_counter"]),
    )
