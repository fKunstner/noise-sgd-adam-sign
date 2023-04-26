import numpy as np
import explib
from explib.expmaker import slurm_configs


def merge_grids(*grids):
    return sorted(list(set.union(*[set(grid) for grid in grids])))


EXPERIMENTS = []

EXPERIMENTS_ADAM = [
    {
        "dataset": dataset,
        "model": "distilbert_base_pretrained",
        "batch_size": bs,
        "max_epoch": 0,
        "seed": seed,
        "model_args": {
            "tgt_len": 384,
            "doc_stride": 128,
        },
        "opt": {"name": "SGD", "alpha": alpha, "momentum": 0.0},
        "init_noise_norm": True,
        "save_norm_samples": True,
        "slurm_config": slurm_configs.LARGE_GPU_6H,
    }
    for dataset in ["squad"]
    for alpha in [1e-2]
    for seed in range(5)
    for bs in [1, 16]
]

EXPERIMENTS.extend(EXPERIMENTS_ADAM)

if __name__ == "__main__":
    explib.expmaker.experiment_maker_cli(
        exp_name="distillbert_squad_noise_hists",
        descr="distill squad norm",
        experiments=EXPERIMENTS,
        as_one_job=True,
    )
