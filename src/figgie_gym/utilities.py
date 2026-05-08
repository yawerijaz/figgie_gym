# pyright: reportPrivateImportUsage=false

import subprocess
from typing import Any, cast

import structlog
import torch

logger = structlog.get_logger(__name__)


def flatten[T](expandable: list[T] | dict[str, T] | T) -> dict[str, Any]:
    if isinstance(expandable, list):
        expandable = cast("list[T]", expandable)
        expandable = {f"item_{i}": vi for i, vi in enumerate(expandable)}

    if isinstance(expandable, dict):
        expandable = cast("dict[str, T]", expandable)
        result: dict[str, T] = {}
        for k, v in expandable.items():
            flat = flatten(v)
            for fk, fv in flat.items():
                result[f"{k}.{fk}"] = fv
        return result
    return {"value": expandable}


def identity[T](x: T) -> T:
    return x


def soft_clip(
    raw_price: torch.Tensor,
    upper: float = 10,
    strength: float = 1,
) -> torch.Tensor:
    return upper * (1 - torch.exp(-raw_price / (strength * upper)))  # pyright: ignore[reportPrivateImportUsage]


def get_git_sha() -> str | None:
    try:
        command = ["git", "rev-parse", "HEAD"]
        result = subprocess.check_output(  # noqa: S603
            command,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.strip()
    except subprocess.CalledProcessError as e:
        logger.warning("Error executing git command", error=e)
        return None
    except FileNotFoundError:
        logger.warning("Git is not installed or not in the system's PATH.")
        return None


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
