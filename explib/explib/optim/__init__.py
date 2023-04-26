"""Optimizers

Generic interface to build optimizers by name,
possibly interfacing with pytorch

"""
import json

import torch
from .signum import Signum
from .modified_adam import ModifiedAdam
from .normalized_gd import (
    PlainSGD,
    NormalizedSGD,
    BlockNormalizedSGD,
    SignSGD,
    RescaledSignDescent,
)
from .clipped_sgd import ClippedGD

SGD = "SGD"
ADAM = "Adam"
ADAM_ABLATION = "AdamAblation"
SIGNUM = "Signum"
PLAIN_SGD = "PlainSGD"
NORMALIZED_GD = "NormalizedGD"
BLOCK_NORMALIZED_GD = "BlockNormalizedGD"
SIGN_D = "SignDescent"
RESCALED_SIGN_D = "RescaledSignDescent"
CLIPPED_SGD = "ClippedGD"

AVAILABLE_OPTIMIZERS = [
    SGD,
    ADAM,
    SIGNUM,
    ADAM_ABLATION,
    NORMALIZED_GD,
    BLOCK_NORMALIZED_GD,
    SIGN_D,
    RESCALED_SIGN_D,
    CLIPPED_SGD,
]


def init(params, model):
    name = params["name"]
    momentum = params["momentum"] if "momentum" in params else 0

    if name not in AVAILABLE_OPTIMIZERS:
        raise Exception("Optimizer {} not available".format(name))

    if name == SGD:
        return torch.optim.SGD(
            model.parameters(), lr=params["alpha"], momentum=momentum
        )

    if name == ADAM:
        return torch.optim.Adam(
            model.parameters(),
            lr=params["alpha"],
            betas=(params["b1"], params["b2"]),
        )

    if name == ADAM_ABLATION:
        params_ = json.loads(json.dumps(params))

        lr = params_.get("alpha")
        betas = (params_.get("b1"), params_.get("b2"))

        params_.pop("name")
        params_.pop("alpha")
        params_.pop("b1")
        params_.pop("b2")

        return ModifiedAdam(model.parameters(), lr=lr, betas=betas, **params_)

    if name == SIGNUM:
        return Signum(model.parameters(), lr=params["alpha"], momentum=momentum)

    if name == PLAIN_SGD:
        return PlainSGD(model.parameters(), lr=params["alpha"], momentum=momentum)
    if name == NORMALIZED_GD:
        return NormalizedSGD(model.parameters(), lr=params["alpha"], momentum=momentum)
    if name == BLOCK_NORMALIZED_GD:
        return BlockNormalizedSGD(
            model.parameters(), lr=params["alpha"], momentum=momentum
        )
    if name == SIGN_D:
        return SignSGD(model.parameters(), lr=params["alpha"], momentum=momentum)
    if name == RESCALED_SIGN_D:
        return RescaledSignDescent(
            model.parameters(), lr=params["alpha"], momentum=momentum
        )

    if name == CLIPPED_SGD:
        return ClippedGD(
            model.parameters(),
            lr=params["alpha"],
            momentum=momentum,
            clipat=params.get("clipat", 0.5),
        )
