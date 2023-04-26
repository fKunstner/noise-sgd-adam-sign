import explib
from explib.expmaker import PROB_PTB_TENC_DET as PTB_TEC
from explib.expmaker import PROB_WT2_TXL_DET as WT2_TXL
from explib.expmaker import merge_dicts, merge_sets, nice_logspace
from explib.expmaker.slurm_configs import DEFAULT_GPU_12H, DEFAULT_GPU_16H
from explib.optim import NORMALIZED_GD, RESCALED_SIGN_D, SIGN_D

hyperparam_names = [
    "dataset",
    "model",
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


def sgd(stepsize, momentum=True):  # his trickle
    return {
        "opt": {
            "name": "SGD",
            "alpha": stepsize,
            "momentum": 0.9 if momentum else 0.0,
        }
    }


SEEDS = [0, 1, 2]


alphas_ptb_adam = nice_logspace(start=-5, end=-2, base=10, density=1)
alphas_ptb_sgd = nice_logspace(start=-3, end=0, base=10, density=1)
alphas_wt2_sgd = nice_logspace(start=-5, end=0, base=10, density=1)
alphas_wt2_adam = merge_sets(
    nice_logspace(start=-6, end=-1, base=10, density=1),
)


optimizers_ptb = (
    [adam(alpha, momentum=True) for alpha in alphas_ptb_adam]
    + [adam(alpha, momentum=False) for alpha in alphas_ptb_adam]
    + [sgd(alpha, momentum=False) for alpha in alphas_ptb_sgd]
    + [sgd(alpha, momentum=True) for alpha in alphas_ptb_sgd]
)
optimizers_wt2 = (
    [adam(alpha, momentum=True) for alpha in alphas_wt2_adam]
    + [adam(alpha, momentum=False) for alpha in alphas_wt2_adam]
    + [sgd(alpha, momentum=False) for alpha in alphas_wt2_sgd]
    + [sgd(alpha, momentum=True) for alpha in alphas_wt2_sgd]
)

base_alphas = nice_logspace(start=-6, end=1, base=10, density=0)
base_alphas_RSD = base_alphas

alphas_for_dset_opt = {
    "ptb": {
        NORMALIZED_GD: nice_logspace(start=-1, end=1, base=10, density=1),
        SIGN_D: nice_logspace(start=-5, end=-2, base=10, density=1),
        RESCALED_SIGN_D: nice_logspace(start=-6, end=-3, base=10, density=1),
    },
    "wt2": {
        NORMALIZED_GD: nice_logspace(start=-1, end=1, base=10, density=1),
        SIGN_D: nice_logspace(start=-5, end=-2, base=10, density=1),
        RESCALED_SIGN_D: nice_logspace(start=-9, end=-3, base=10, density=1),
    },
}
alphas_for_dset_opt_with_momentum = {
    "ptb": {
        NORMALIZED_GD: nice_logspace(start=-2, end=0, base=10, density=1),
        SIGN_D: nice_logspace(start=-5, end=-2, base=10, density=1),
        RESCALED_SIGN_D: nice_logspace(start=-8, end=-5, base=10, density=1),
    },
    "wt2": {
        NORMALIZED_GD: nice_logspace(start=-2, end=0, base=10, density=1),
        SIGN_D: nice_logspace(start=-5, end=-2, base=10, density=1),
        RESCALED_SIGN_D: nice_logspace(start=-9, end=-6, base=10, density=1),
    },
}

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
    for k in ["ptb", "wt2"]
}

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


EXPERIMENTS = (
    [
        merge_dicts(PTB_TEC, size_settings, opt_settings, {"seed": seed})
        for size_settings in settings_ptb
        for opt_settings in optimizers_ptb
        for seed in SEEDS
    ]
    + [
        merge_dicts(WT2_TXL, size_settings, opt_settings, {"seed": seed})
        for size_settings in settings_wt2
        for opt_settings in optimizers_wt2
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
)


if __name__ == "__main__":
    explib.expmaker.experiment_maker_cli(
        exp_name="no-dropout",
        descr="Repeat of the same experiments without dropout",
        experiments=EXPERIMENTS,
        hyperparam_names=hyperparam_names,
        as_job_array=True,
    )
