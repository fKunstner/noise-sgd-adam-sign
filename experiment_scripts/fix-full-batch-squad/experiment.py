"""Sanity checks for the full runs, checking runtime and memory consumption of
various configurations-"""

import explib
from explib.expmaker import PROB_DB_SQD as DB_SQD
from explib.expmaker import merge_dicts, nice_logspace
from explib.expmaker.slurm_configs import DEFAULT_GPU_36H, LARGE_GPU_36H
from explib.optim import NORMALIZED_GD, SIGN_D

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

settings_BRT = {
    "batch_size": 16,
    "slurm_config": LARGE_GPU_36H,
    "accumulate_steps": 1370 * 4,
    "max_epoch": 80,
    "drop_last": True,
    "shuffle": False,
}


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
        "opt": {"name": "SGD", "alpha": stepsize, "momentum": 0.9 if momentum else 0.0}
    }


def normgd(stepsize, momentum=True):
    return {
        "opt": {
            "name": NORMALIZED_GD,
            "alpha": stepsize,
            "momentum": 0.9 if momentum else 0,
        }
    }


def signd(stepsize, momentum=True):
    return {
        "opt": {"name": SIGN_D, "alpha": stepsize, "momentum": 0.9 if momentum else 0}
    }


alphas_BRT_sgd = nice_logspace(start=-2, end=0, base=10, density=1)
alphas_BRT_sgd_m = nice_logspace(start=-2, end=-1, base=10, density=1)
alphas_BRT_adam = nice_logspace(start=-4, end=-3, base=10, density=1)
alphas_BRT_adam_m = nice_logspace(start=-4, end=-1, base=10, density=1)
alphas_BRT_NormalizedGD = nice_logspace(start=-1, end=0, base=10, density=1)
alphas_BRT_SignDescent = nice_logspace(start=-5, end=-3, base=10, density=1)
alphas_BRT_NormalizedGD_m = nice_logspace(start=-2, end=0, base=10, density=1)
alphas_BRT_SignDescent_m = nice_logspace(start=-6, end=-2, base=10, density=1)

opts_BRT = (
    [sgd(alpha, False) for alpha in alphas_BRT_sgd]
    + [adam(alpha, False) for alpha in alphas_BRT_adam]
    + [normgd(alpha, False) for alpha in alphas_BRT_NormalizedGD]
    + [signd(alpha, False) for alpha in alphas_BRT_SignDescent]
    + [sgd(alpha, True) for alpha in alphas_BRT_sgd_m]
    + [adam(alpha, True) for alpha in alphas_BRT_adam_m]
    + [normgd(alpha, True) for alpha in alphas_BRT_NormalizedGD_m]
    + [signd(alpha, True) for alpha in alphas_BRT_SignDescent_m]
)

EXPERIMENTS = [
    merge_dicts(DB_SQD, settings_BRT, opt_settings, {"seed": seed})
    for opt_settings in opts_BRT
    for seed in [0, 1, 2]
]


if __name__ == "__main__":
    explib.expmaker.experiment_maker_cli(
        exp_name="fix-full-batch-training-squad",
        descr="Rerun of experiments on Squad with shuffle=False",
        experiments=EXPERIMENTS,
        hyperparam_names=hyperparam_names,
        as_job_array=True,
    )
