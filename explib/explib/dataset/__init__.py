"""Datasets.

General interface to load a dataset
"""
import os
from pathlib import Path

from explib import config

from .cifar_loader import cifar_loader
from .glue_loader import glue_loader
from .language_loader import ptb_loader, wikitext2_loader
from .squad_loader import squad_loader
from .torchvision_loader import torchvision_loader

MNIST = "mnist"
WIKITEXT2 = "wikitext2"
CIFAR10 = "cifar10"
CIFAR100 = "cifar100"
PTB = "ptb"
SQUAD = "squad"
ADVERSARIAL_QA = "adversarial_qa"
GLUE = "glue"

AVAILABLE_DATASET = [
    MNIST,
    WIKITEXT2,
    CIFAR10,
    CIFAR100,
    PTB,
    SQUAD,
    #    GLUE,
]


def init(
    dataset_name,
    batch_size,
    device,
    extra_params=None,
    drop_last=False,
    fake_full_batch_mode=False,
    accelerator=None,
    shuffle=True,
    outliers_filename=None,
):
    extra_params = extra_params if extra_params is not None else {}
    dataset_path = os.path.join(config.get_workspace(), "datasets")
    Path(dataset_path).mkdir(parents=True, exist_ok=True)

    if fake_full_batch_mode and dataset_name not in [CIFAR10, CIFAR100, SQUAD]:
        raise NotImplementedError(
            "Fake full batch mode not implemented for {dataset_name}"
        )

    if dataset_name == MNIST:
        return torchvision_loader(
            dataset_name, batch_size, drop_last=drop_last, shuffle=shuffle
        )
    elif dataset_name == WIKITEXT2:
        return wikitext2_loader(
            batch_size,
            device,
            extra_params.get("tgt_len", 128),
            drop_last=drop_last,
        )
    elif dataset_name == CIFAR10:
        return cifar_loader(
            batch_size,
            drop_last=drop_last,
            fake_full_batch_mode=fake_full_batch_mode,
            shuffle=shuffle,
        )
    elif dataset_name == CIFAR100:
        return cifar_loader(
            batch_size,
            load_100=True,
            drop_last=drop_last,
            fake_full_batch_mode=fake_full_batch_mode,
        )
    elif dataset_name == PTB:
        return ptb_loader(
            batch_size,
            device,
            extra_params.get("tgt_len", 128),
            drop_last=drop_last,
            outliers_filename=outliers_filename,
        )
    elif dataset_name == SQUAD or dataset_name == ADVERSARIAL_QA:
        return squad_loader(
            dataset_name,
            batch_size,
            extra_params.get("tgt_len", 384),
            extra_params.get("doc_stride", 128),
            model_name=extra_params.get("model_name", "bert_base_pretrained"),
            drop_last=drop_last,
            fake_full_batch_mode=fake_full_batch_mode,
            shuffle=shuffle,
            outliers_filename=outliers_filename,
        )
    else:
        raise Exception("Dataset {} not available".format(dataset_name))
