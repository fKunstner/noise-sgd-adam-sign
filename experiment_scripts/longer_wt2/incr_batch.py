"""Sanity checks for the full runs, checking runtime and memory consumption of
various configurations-"""

import explib
from explib.expmaker.slurm_configs import (
    SMALL_GPU_4H,
    SMALL_GPU_12H,
    SMALL_GPU_8H,
    LARGE_GPU_24H,
    LARGE_GPU_12H,
)
from explib.expmaker import merge_dicts, nice_logspace, merge_sets

from explib.expmaker import (
    PROB_MNIST_LENET5 as MNI_LN5,
    PROB_CIFAR10_RESNET18 as C10_R18,
    PROB_PTB_TENC as PTB_TEC,
    PROB_DB_SQD as DB_SQD,
    PROB_WT2_TXL as WT2_TXL,
)

hyperparam_names = [
    "dataset",
    "batch_size",
    "opt.name",
    "opt.b1",
    "opt.momentum",
    "accumulate_steps",
    "seed",
    "opt.alpha",
    "slurm_config",
]


def adam(stepsize, momentum=True):
    return {
        "opt": {
            "name": "Adam",
            "alpha": stepsize,
            "b1": 0.9 if momentum else 0.0,
            "b2": 0.999,
        }
    }


def sgd(stepsize, momentum=True):
    return {
        "opt": {
            "name": "SGD",
            "alpha": stepsize,
            "momentum": 0.9 if momentum else 0.0,
        }
    }


SEEDS = [0, 1, 2]

alphas_wt2_sgd = nice_logspace(start=-5, end=0, base=10, density=1)
alphas_wt2_adam = merge_sets(
    nice_logspace(start=-6, end=-1, base=10, density=1),
    nice_logspace(start=-4, end=-2, base=10, density=2),
)

optimizers_wt2 = (
    [adam(alpha, momentum=True) for alpha in alphas_wt2_adam]
    + [adam(alpha, momentum=False) for alpha in alphas_wt2_adam]
    + [sgd(alpha, momentum=False) for alpha in alphas_wt2_sgd]
    + [sgd(alpha, momentum=True) for alpha in alphas_wt2_sgd]
)


def settings_wt2_(bs=32, accum=1, slurm=SMALL_GPU_4H, epoch=5):
    return {
        "batch_size": bs,
        "accumulate_steps": accum,
        "slurm_config": slurm,
        "max_epoch": epoch,
    }


settings_wt2 = [
    settings_wt2_(bs=20, accum=1, slurm=SMALL_GPU_4H, epoch=40),
    settings_wt2_(bs=80, accum=1, slurm=SMALL_GPU_4H, epoch=40),
    settings_wt2_(bs=80, accum=4, slurm=SMALL_GPU_4H, epoch=80),
    settings_wt2_(bs=80, accum=16, slurm=SMALL_GPU_8H, epoch=160),
]

EXPERIMENTS = +[
    merge_dicts(WT2_TXL, size_settings, opt_settings, {"seed": seed})
    for size_settings in settings_wt2
    for opt_settings in optimizers_wt2
    for seed in SEEDS
]


if __name__ == "__main__":
    explib.expmaker.experiment_maker_cli(
        exp_name="increasing-batch-size",
        descr="Increasing batch size experiments",
        experiments=EXPERIMENTS,
        hyperparam_names=hyperparam_names,
        as_job_array=True,
    )
