import numpy as np
import explib
from explib.expmaker import slurm_configs


def merge_grids(*grids):
    return sorted(list(set.union(*[set(grid) for grid in grids])))


EXPERIMENTS = []


EXPERIMENTS_MNIST = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "mnist",
        "model": "lenet5",
        "batch_size": bs,
        "max_epoch": 0,
        "init_noise_norm": True,
        "seed": seed,
        "slurm_config": slurm_configs.SMALL_GPU_2H,
        "opt": {
            "name": "Adam",
            "alpha": 0.001,
            "b1": 0.99,
            "b2": 0.999,
        },
    }
    for bs in [1, 256]
    for seed in range(5)
]

EXPERIMENTS.extend(EXPERIMENTS_MNIST)

EXPERIMENTS_RESNET18 = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "cifar10",
        "model": "resnet18",
        "batch_size": bs,
        "max_epoch": 0,
        "init_noise_norm": True,
        "seed": seed,
        "slurm_config": slurm_configs.SMALL_GPU_2H,
        "opt": {
            "name": "Adam",
            "alpha": 0.001,
            "b1": 0.99,
            "b2": 0.999,
        },
    }
    for bs in [2, 64]
    for seed in range(5)
]

EXPERIMENTS.extend(EXPERIMENTS_RESNET18)

if __name__ == "__main__":
    explib.expmaker.experiment_maker_cli(
        exp_name="image_hists",
        descr="image problem noise histograms",
        as_one_job=True,
        experiments=EXPERIMENTS,
    )
