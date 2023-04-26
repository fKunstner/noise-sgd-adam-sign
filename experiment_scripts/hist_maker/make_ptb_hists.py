import numpy as np
import explib
from explib.expmaker import slurm_configs


def merge_grids(*grids):
    return sorted(list(set.union(*[set(grid) for grid in grids])))


EXPERIMENTS = []


EXPERIMENTS_SGD = [
    {
        "loss_func": "logloss",
        "dataset": "ptb",
        "model": "transformer_encoder",
        "model_args": {
            "tgt_len": 35,
        },
        "batch_size": bs,
        "max_epoch": 0,
        "seed": seed,
        "opt": {
            "name": "Adam",
            "alpha": 0.001,
            "b1": 0.99,
            "b2": 0.999,
        },
        "init_noise_norm": True,
        "save_norm_samples": True,
        "slurm_config": slurm_configs.SMALL_GPU_2H,
    }
    for bs in [1, 16]
    for seed in range(5)
]

EXPERIMENTS.extend(EXPERIMENTS_SGD)

if __name__ == "__main__":
    explib.expmaker.experiment_maker_cli(
        exp_name="ptb_noise_hists",
        descr="ptb noise histograms",
        as_one_job=True,
        experiments=EXPERIMENTS,
    )
