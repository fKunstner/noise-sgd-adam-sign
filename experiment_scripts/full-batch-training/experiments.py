"""
Sanity checks for the full runs,
checking runtime and memory consumption of various configurations-
"""

import explib
from explib.expmaker.slurm_configs import (
    DEFAULT_GPU_12H,
    LARGE_GPU_24H,
    DEFAULT_GPU_16H,
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

alphas_mnist_adam = nice_logspace(start=-4, end=0, base=10, density=1)
alphas_mnist_sgd = nice_logspace(start=-3, end=1, base=10, density=1)
alphas_cifar_adam = nice_logspace(start=-6, end=0, base=10, density=1)
alphas_cifar_sgd = nice_logspace(start=-5, end=1, base=10, density=1)
alphas_ptb_adam = nice_logspace(start=-5, end=-2, base=10, density=1)
alphas_ptb_sgd = nice_logspace(start=-3, end=0, base=10, density=1)
alphas_wt2_sgd = nice_logspace(start=-5, end=0, base=10, density=1)
alphas_wt2_adam = merge_sets(
    nice_logspace(start=-6, end=-1, base=10, density=1),
    nice_logspace(start=-4, end=-2, base=10, density=2),
)
alphas_squad_adam = nice_logspace(start=-6, end=-2, base=10, density=1)
alphas_squad_sgd = nice_logspace(start=-4, end=0, base=10, density=1)


optimizers = (
    [adam(alpha, momentum=True) for alpha in alphas_mnist_adam]
    + [adam(alpha, momentum=False) for alpha in alphas_mnist_adam]
    + [sgd(alpha, momentum=False) for alpha in alphas_mnist_sgd]
    + [sgd(alpha, momentum=True) for alpha in alphas_mnist_sgd]
)
optimizers_cifar = (
    [adam(alpha, momentum=True) for alpha in alphas_cifar_adam]
    + [adam(alpha, momentum=False) for alpha in alphas_cifar_adam]
    + [sgd(alpha, momentum=False) for alpha in alphas_cifar_sgd]
    + [sgd(alpha, momentum=True) for alpha in alphas_cifar_sgd]
)
optimizers_ptb = (
    [adam(alpha, momentum=True) for alpha in alphas_ptb_adam]
    + [adam(alpha, momentum=False) for alpha in alphas_ptb_adam]
    + [sgd(alpha, momentum=False) for alpha in alphas_ptb_sgd]
    + [sgd(alpha, momentum=True) for alpha in alphas_ptb_sgd]
)
optimizers_squad = (
    [adam(alpha, momentum=True) for alpha in alphas_squad_adam]
    + [adam(alpha, momentum=False) for alpha in alphas_squad_adam]
    + [sgd(alpha, momentum=False) for alpha in alphas_squad_sgd]
    + [sgd(alpha, momentum=True) for alpha in alphas_squad_sgd]
)
optimizers_wt2 = (
    [adam(alpha, momentum=True) for alpha in alphas_wt2_adam]
    + [adam(alpha, momentum=False) for alpha in alphas_wt2_adam]
    + [sgd(alpha, momentum=False) for alpha in alphas_wt2_sgd]
    + [sgd(alpha, momentum=True) for alpha in alphas_wt2_sgd]
)

settings_mnist = [
    {
        "batch_size": 20000,
        "slurm_config": DEFAULT_GPU_12H,
        "accumulate_steps": 3,
        "max_epoch": 800,
    },
]

settings_cifar = [
    {
        "batch_size": 10000,
        "slurm_config": DEFAULT_GPU_12H,
        "accumulate_steps": 5,
        "max_epoch": 800,
    },
]

settings_ptb = [
    {
        "batch_size": 1326,
        "slurm_config": DEFAULT_GPU_12H,
        "accumulate_steps": 20,
        "max_epoch": 800 * 4,
        "drop_last": True,
    },
]

settings_squad = [
    {
        "batch_size": 64,
        "slurm_config": LARGE_GPU_24H,
        "accumulate_steps": 1370,
        "max_epoch": 20,
        "drop_last": True,
    },
]

settings_wt2 = [
    {
        "batch_size": 80,
        "accumulate_steps": 203,
        "slurm_config": DEFAULT_GPU_16H,
        "max_epoch": 320,
        "drop_last": True,
    }
]


EXPERIMENTS = (
    [
        merge_dicts(MNI_LN5, size_settings, opt_settings, {"seed": seed})
        for size_settings in settings_mnist
        for opt_settings in optimizers
        for seed in SEEDS
    ]
    + [
        merge_dicts(C10_R18, size_settings, opt_settings, {"seed": seed})
        for size_settings in settings_cifar
        for opt_settings in optimizers_cifar
        for seed in SEEDS
    ]
    + [
        merge_dicts(PTB_TEC, size_settings, opt_settings, {"seed": seed})
        for size_settings in settings_ptb
        for opt_settings in optimizers_ptb
        for seed in SEEDS
    ]
    + [
        merge_dicts(DB_SQD, size_settings, opt_settings, {"seed": seed})
        for size_settings in settings_squad
        for opt_settings in optimizers_squad
        for seed in SEEDS
    ]
    + [
        merge_dicts(WT2_TXL, size_settings, opt_settings, {"seed": seed})
        for size_settings in settings_wt2
        for opt_settings in optimizers_wt2
        for seed in SEEDS
    ]
)


if __name__ == "__main__":
    explib.expmaker.experiment_maker_cli(
        exp_name="full-batch-training",
        descr="Full batch training on the standard datasets",
        experiments=EXPERIMENTS,
        hyperparam_names=hyperparam_names,
        as_job_array=True,
    )
