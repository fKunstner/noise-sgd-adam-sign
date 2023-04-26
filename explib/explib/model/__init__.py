from .letnet5 import LeNet5
from .linear_model import LinearModel
from .transformer_encoder import TransformerEncoderModel
from .resnet import getResNet
from .full_connected import FullyConnected
from .transformer_xl import MemTransformerLM
from .bert_base_pretrained import (
    get_bert_base_pretrained,
    get_distilbert_base_pretrained,
)
from .bert_glue import get_bert_glue
from ..util import weights_init

LENET5 = "lenet5"
LIN_REG = "lin_reg"
LOG_REG = "log_reg"
TRANSFORMER_ENCODER = "transformer_encoder"
TRANSFORMER_ENCODER_DET = "transformer_encoder_deterministic"
RESNET50 = "resnet50"
RESNET34 = "resnet34"
RESNET18 = "resnet18"
RESNET101 = "resnet101"
FULLY_CONNECTED = "fc"
TRANSFORMER_XL = "transformer_xl"
TRANSFORMER_XL_DET = "transformer_xl_deterministic"
BERT_BASE = "bert_base_pretrained"
BERT_GLUE = "bert_base_cased"
DISTILBERT = "distilbert_base_pretrained"

AVAILABLE_MODELS = [
    LENET5,
    LIN_REG,
    LOG_REG,
    TRANSFORMER_ENCODER,
    TRANSFORMER_ENCODER_DET,
    RESNET101,
    RESNET50,
    RESNET34,
    RESNET18,
    FULLY_CONNECTED,
    TRANSFORMER_XL,
    TRANSFORMER_XL_DET,
    BERT_BASE,
    BERT_GLUE,
    DISTILBERT,
]


def init(model_name, model_args=None, features_dim=0, transformer_len=0):
    if model_name == LENET5:
        if model_args is not None:
            return LeNet5(10, in_channels=model_args["in_channels"])
        return LeNet5(10)

    elif model_name == LIN_REG:
        return LinearModel(features_dim, 1)

    elif model_name == LOG_REG:
        return LinearModel(features_dim, 2)

    elif model_name == TRANSFORMER_ENCODER:
        model = TransformerEncoderModel(transformer_len, 200, 2, 200, 2, 0.2)
        model.apply(weights_init)
        return model

    elif model_name == TRANSFORMER_ENCODER_DET:
        model = TransformerEncoderModel(transformer_len, 200, 2, 200, 2, dropout=0.0)
        model.apply(weights_init)
        return model

    elif model_name == RESNET50:
        return getResNet(50)

    elif model_name == RESNET34:
        return getResNet(34)

    elif model_name == RESNET18:
        return getResNet(18)

    elif model_name == RESNET101:
        return getResNet(101)

    elif model_name == FULLY_CONNECTED:
        return FullyConnected()

    elif model_name == TRANSFORMER_XL:
        model = MemTransformerLM(
            transformer_len,
            model_args["n_layer"],
            model_args["n_head"],
            model_args["d_model"],
            model_args["d_head"],
            model_args["d_inner"],
            model_args["dropout"],
            model_args["dropatt"],
            tie_weight=False,
            d_embed=model_args["d_model"],
            tgt_len=model_args["tgt_len"],
            ext_len=0,
            mem_len=model_args["mem_len"],
            same_length=False,
        )
        model.apply(weights_init)
        model.word_emb.apply(weights_init)
        return model

    elif model_name == TRANSFORMER_XL_DET:
        model = MemTransformerLM(
            transformer_len,
            model_args["n_layer"],
            model_args["n_head"],
            model_args["d_model"],
            model_args["d_head"],
            model_args["d_inner"],
            dropout=0,
            dropatt=0,
            tie_weight=False,
            d_embed=model_args["d_model"],
            tgt_len=model_args["tgt_len"],
            ext_len=0,
            mem_len=model_args["mem_len"],
            same_length=False,
        )
        model.apply(weights_init)
        model.word_emb.apply(weights_init)
        return model

    elif model_name == BERT_BASE:
        return get_bert_base_pretrained()
    elif model_name == DISTILBERT:
        return get_distilbert_base_pretrained()
    elif model_name == BERT_GLUE:
        return get_bert_glue(model_args)
    else:
        raise Exception("Model {} not available".format(model_name))
