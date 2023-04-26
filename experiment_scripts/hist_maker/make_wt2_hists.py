import explib
from explib.expmaker import slurm_configs, BASE_PROBLEMS

EXPERIMENTS = [
    {
        **BASE_PROBLEMS["WT2_TRANSFORMERXL"],
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
        "slurm_config": slurm_configs.DEFAULT_GPU_8H,
    }
    for bs in [1, 16]
    for seed in range(5)
]

if __name__ == "__main__":
    explib.expmaker.experiment_maker_cli(
        exp_name="wt2_noise_hist",
        descr="ptb noise histograms",
        as_one_job=True,
        experiments=EXPERIMENTS,
    )
