import itertools

import torch
from torch import Tensor
from torch.optim import SGD
from torch.optim.optimizer import Optimizer, required
from torch.nn.utils import parameters_to_vector as p2v
from typing import List, Optional


class ClippedGD(SGD):
    def __init__(
        self,
        params,
        lr=required,
        clipat=0.5,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
    ):
        if clipat < 0.0:
            raise ValueError("Invalid clipat value: {}".format(clipat))
        self._clipat = clipat
        self.params = params
        super().__init__(
            params,
            lr,
            momentum,
            dampening,
            weight_decay,
            nesterov,
        )

    def step(self, closure=None):
        """Clips the gradients and takes a step of GD. Changes the values of the gradients."""

        torch.nn.utils.clip_grad_norm_(
            itertools.chain(*[group["params"] for group in self.param_groups]),
            max_norm=self._clipat,
        )
        super().step(closure=closure)
