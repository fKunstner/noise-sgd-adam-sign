from ..model import *
from .bert_squad_prob import BertSquadProb
from .image_prob import ImageProb
from .simple_prob import SimpleProb
from .transformer_prob import TransformerProb

image_models = [
    LENET5,
    RESNET18,
    RESNET34,
    RESNET50,
    RESNET101,
]

simple_models = [
    LIN_REG,
    LOG_REG,
    FULLY_CONNECTED,
]

transformer_models = [
    TRANSFORMER_ENCODER,
    TRANSFORMER_ENCODER_DET,
    TRANSFORMER_XL,
    TRANSFORMER_XL_DET,
]

bert_squad = [BERT_BASE, DISTILBERT]
bert_glue = [BERT_GLUE]


def init(exp_dict):
    model_name = exp_dict["model"]

    if model_name in simple_models:
        return SimpleProb(exp_dict)
    elif model_name in image_models:
        return ImageProb(exp_dict)
    elif model_name in transformer_models:
        return TransformerProb(exp_dict)
    elif model_name in bert_squad:
        return BertSquadProb(exp_dict)

    raise Exception("Model {} not available".format(model_name))
