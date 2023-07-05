from __future__ import annotations

from typing import Any, OrderedDict, TypedDict

import torch


class State(TypedDict):
    """The training state."""

    epoch: int
    model_state_dict: OrderedDict[str, torch.Tensor]
    optim_state_dict: dict[str, Any]
    loss: float
    all_losses: list[float]
