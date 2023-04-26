import os
import random
from explib import config

from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
)


def get_bert_glue(model_args):
    num_labels, task_name = model_args["num_labels"], model_args["task_name"]
    autoconfig = AutoConfig.from_pretrained(
        "bert-base-cased", num_labels=num_labels, finetuning_task=task_name
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased",
        from_tf=False,
        config=autoconfig,
    )

    if "freeze_embedding" in model_args and model_args["freeze_embedding"]:
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False

    if "num_encoder_layers_to_freeze" in model_args:
        num_layers = model_args["num_encoder_layers_to_freeze"]
        for layer in model.bert.encoder.layer[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    return model
