"""Sanity checks for the full runs, checking runtime and memory consumption of
various configurations-"""

import explib
from explib.expmaker import PROB_CIFAR10_RESNET18 as C10_R18
from explib.expmaker import PROB_DB_SQD as DB_SQD
from explib.expmaker import PROB_MNIST_LENET5 as MNI_LN5
from explib.expmaker import PROB_PTB_TENC as PTB_TEC
from explib.expmaker import PROB_WT2_TXL as WT2_TXL
from explib.expmaker import merge_dicts, merge_sets, nice_logspace
from explib.expmaker.slurm_configs import (
    DEFAULT_GPU_12H,
    DEFAULT_GPU_16H,
    DEFAULT_GPU_36H,
    LARGE_GPU_24H,
    LARGE_GPU_36H,
    LARGE_GPU_72H,
)
from explib.optim import (
    BLOCK_NORMALIZED_GD,
    NORMALIZED_GD,
    RESCALED_SIGN_D,
    SGD,
    SIGN_D,
)

hyperparam_names = [
    "dataset",
    "batch_size",
    "opt.name",
    "accumulate_steps",
    "seed",
    "opt.alpha",
    "slurm_config",
]


SEEDS = [0, 1, 2]
base_alphas = nice_logspace(start=-6, end=1, base=10, density=0)
base_alphas_RSD = base_alphas
# base_alphas_RSD = nice_logspace(start=-10, end=-3, base=10, density=0)

alphas_for_dset_opt = {
    "mnist": {
        NORMALIZED_GD: nice_logspace(start=-2, end=0, base=10, density=1),
        SIGN_D: nice_logspace(start=-4, end=-2, base=10, density=1),
        RESCALED_SIGN_D: nice_logspace(start=-6, end=-1, base=10, density=1),
    },
    "cifar10": {
        NORMALIZED_GD: nice_logspace(start=-2, end=0, base=10, density=1),
        SIGN_D: nice_logspace(start=-6, end=-4, base=10, density=1),
        RESCALED_SIGN_D: nice_logspace(start=-9, end=-5, base=10, density=1),
    },
    "ptb": {
        NORMALIZED_GD: nice_logspace(start=-1, end=1, base=10, density=1),
        SIGN_D: nice_logspace(start=-5, end=-3, base=10, density=1),
        RESCALED_SIGN_D: nice_logspace(start=-6, end=-3, base=10, density=1),
    },
    "wt2": {
        NORMALIZED_GD: nice_logspace(start=-1, end=1, base=10, density=1),
        SIGN_D: nice_logspace(start=-5, end=-3, base=10, density=1),
        RESCALED_SIGN_D: nice_logspace(start=-9, end=-3, base=10, density=1),
    },
    "squad": {
        NORMALIZED_GD: nice_logspace(start=-2, end=0, base=10, density=1),
        SIGN_D: nice_logspace(start=-5, end=-3, base=10, density=1),
        RESCALED_SIGN_D: nice_logspace(start=-7, end=-5, base=10, density=1),
    },
}
alphas_for_dset_opt_with_momentum = {
    "mnist": {
        NORMALIZED_GD: nice_logspace(start=-2, end=0, base=10, density=1),
        SIGN_D: nice_logspace(start=-5, end=-3, base=10, density=1),
        RESCALED_SIGN_D: nice_logspace(start=-6, end=-3, base=10, density=1),
    },
    "cifar10": {
        NORMALIZED_GD: nice_logspace(start=-3, end=0, base=10, density=1),
        SIGN_D: nice_logspace(start=-8, end=-5, base=10, density=1),
        RESCALED_SIGN_D: nice_logspace(start=-8, end=-4, base=10, density=1),
    },
    "ptb": {
        NORMALIZED_GD: nice_logspace(start=-2, end=0, base=10, density=1),
        SIGN_D: nice_logspace(start=-5, end=-3, base=10, density=1),
        RESCALED_SIGN_D: nice_logspace(start=-8, end=-5, base=10, density=1),
    },
    "wt2": {
        NORMALIZED_GD: nice_logspace(start=-2, end=0, base=10, density=1),
        SIGN_D: nice_logspace(start=-5, end=-3, base=10, density=1),
        RESCALED_SIGN_D: nice_logspace(start=-9, end=-6, base=10, density=1),
    },
    "squad": {
        NORMALIZED_GD: nice_logspace(start=-2, end=0, base=10, density=1),
        SIGN_D: nice_logspace(start=-3, end=-0, base=10, density=1),
        RESCALED_SIGN_D: nice_logspace(start=-11, end=-8, base=10, density=1),
    },
}


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


settings_wt2 = [
    {
        "batch_size": 80,
        "accumulate_steps": 203,
        "slurm_config": DEFAULT_GPU_16H,
        "max_epoch": 320,
        "drop_last": True,
    }
]


settings_squad = [
    #    {
    #        "batch_size": 16,
    #        "slurm_config": DEFAULT_GPU_24H,
    #        "accumulate_steps": 1370 * 4,
    #        "max_epoch": 20,
    #        "drop_last": True,
    #        "shuffle": False,
    #    },
    {
        "batch_size": 16,
        "slurm_config": LARGE_GPU_36H,
        "accumulate_steps": 1370 * 4,
        "max_epoch": 80,
        "drop_last": True,
    },
]

base_optims = (
    [
        {"opt": {"name": name, "alpha": alpha}}
        for name in [NORMALIZED_GD, SIGN_D]
        for alpha in base_alphas
    ]
    + [
        {"opt": {"name": name, "alpha": alpha, "momentum": 0.9}}
        for name in [NORMALIZED_GD, SIGN_D]
        for alpha in base_alphas
    ]
    + [
        {"opt": {"name": name, "alpha": alpha, "norm": 1}}
        for name in [RESCALED_SIGN_D]
        for alpha in base_alphas_RSD
    ]
    + [
        {"opt": {"name": name, "alpha": alpha, "momentum": 0.9, "norm": 1}}
        for name in [RESCALED_SIGN_D]
        for alpha in base_alphas_RSD
    ]
)

optimizers_for_dataset = {
    k: [
        {"opt": {"name": name, "alpha": alpha}}
        for name in [NORMALIZED_GD, SIGN_D]
        for alpha in alphas_for_dset_opt[k][name]
    ]
    + [
        {"opt": {"name": name, "alpha": alpha, "momentum": 0.9}}
        for name in [NORMALIZED_GD, SIGN_D]
        for alpha in alphas_for_dset_opt_with_momentum[k][name]
    ]
    + [
        {"opt": {"name": name, "alpha": alpha, "norm": 1}}
        for name in [RESCALED_SIGN_D]
        for alpha in alphas_for_dset_opt[k][name]
    ]
    + [
        {"opt": {"name": name, "alpha": alpha, "momentum": 0.9, "norm": 1}}
        for name in [RESCALED_SIGN_D]
        for alpha in alphas_for_dset_opt_with_momentum[k][name]
    ]
    for k in ["mnist", "cifar10", "ptb", "wt2", "squad"]
}

EXPERIMENTS = (
    [
        merge_dicts(MNI_LN5, size_settings, opt_settings, {"seed": 0})
        for size_settings in settings_mnist
        for opt_settings in base_optims
    ]
    + [
        merge_dicts(C10_R18, size_settings, opt_settings, {"seed": 0})
        for size_settings in settings_cifar
        for opt_settings in base_optims
    ]
    + [
        merge_dicts(PTB_TEC, size_settings, opt_settings, {"seed": 0})
        for size_settings in settings_ptb
        for opt_settings in base_optims
    ]
    + [
        merge_dicts(WT2_TXL, size_settings, opt_settings, {"seed": 0})
        for size_settings in settings_wt2
        for opt_settings in base_optims
    ]
    + [
        merge_dicts(DB_SQD, size_settings, opt_settings, {"seed": 0})
        for size_settings in settings_squad
        for opt_settings in base_optims
    ]
    + [
        merge_dicts(MNI_LN5, size_settings, opt_settings, {"seed": seed})
        for size_settings in settings_mnist
        for opt_settings in optimizers_for_dataset["mnist"]
        for seed in SEEDS
    ]
    + [
        merge_dicts(C10_R18, size_settings, opt_settings, {"seed": seed})
        for size_settings in settings_cifar
        for opt_settings in optimizers_for_dataset["cifar10"]
        for seed in SEEDS
    ]
    + [
        merge_dicts(PTB_TEC, size_settings, opt_settings, {"seed": seed})
        for size_settings in settings_ptb
        for opt_settings in optimizers_for_dataset["ptb"]
        for seed in SEEDS
    ]
    + [
        merge_dicts(WT2_TXL, size_settings, opt_settings, {"seed": seed})
        for size_settings in settings_wt2
        for opt_settings in optimizers_for_dataset["wt2"]
        for seed in SEEDS
    ]
    + [
        merge_dicts(DB_SQD, size_settings, opt_settings, {"seed": seed})
        for size_settings in settings_squad
        for opt_settings in optimizers_for_dataset["squad"]
        for seed in SEEDS
    ]
)

EXPERIMENTS = explib.expmaker.make_exp_dict_list_unique(EXPERIMENTS)

if __name__ == "__main__":
    explib.expmaker.experiment_maker_cli(
        exp_name="full-batch-training-normalized-optimizers",
        descr="Full batch training with normalization-based optimizers (Sign, Rescaled Sign, Normalized GD)",
        experiments=EXPERIMENTS,
        hyperparam_names=hyperparam_names,
        as_job_array=True,
    )
