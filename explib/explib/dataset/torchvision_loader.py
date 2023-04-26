import os

import torch
import torchvision
from explib import config
from torchvision import transforms
from torchvision.datasets import MNIST, USPS


def torchvision_loader(dataset_name, batch_size, drop_last=False, shuffle=True):
    if dataset_name == "mnist":
        loader = MNIST
    elif dataset_name == "usps":
        loader = USPS
    else:
        raise Exception("Dataset {} not available".format(dataset_name))

    train_dataloader = torch.utils.data.DataLoader(
        loader(
            os.path.join(config.get_workspace(), "datasets"),
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    valid_dataloader = torch.utils.data.DataLoader(
        loader(
            os.path.join(config.get_workspace(), "datasets"),
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
        # drop_last=drop_last,
    )

    return train_dataloader, valid_dataloader
