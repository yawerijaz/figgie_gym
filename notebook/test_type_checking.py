# ruff: noqa: F722  # noqa: INP001
"""`pytest notebook/test_type_checking.py`."""

import jaxtyping
import pytest
import torch
from beartype import beartype
from jaxtyping import (
    Float,
    jaxtyped,  # pyright: ignore[reportUnknownVariableType]
)
from torch import Tensor


@jaxtyped(typechecker=beartype)
def batch_outer_product(
    x: Float[Tensor, "B C"],  # noqa: ARG001
    y: Float[Tensor, "B C"],  # noqa: ARG001
) -> bool:
    return True


def test_ok() -> None:
    batch = 5
    c1 = 2
    c2 = 2
    batch_outer_product(torch.ones((batch, c1)), torch.ones((batch, c2)))  # pyright: ignore[reportPrivateImportUsage]


def test_ko() -> None:
    batch = 5
    c1 = 3
    c2 = 2
    with pytest.raises(jaxtyping.TypeCheckError):
        batch_outer_product(torch.ones((batch, c1)), torch.ones((batch, c2)))  # pyright: ignore[reportPrivateImportUsage]
