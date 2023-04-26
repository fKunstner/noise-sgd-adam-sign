"""Helpers to create experiments."""

import argparse

import numpy as np
from explib import cli_helper, config, logging
from explib.expmaker import wandb_reporting
from explib.expmaker.experiment_defs import (
    create_experiment_definitions,
    exp_dict_from_str,
    exp_dict_to_str,
    get_exp_def_folder,
    get_exp_full_path_json,
    get_job_path,
    get_jobs_folder,
    load_summary,
)
from explib.expmaker.sbatch_writers import (
    create_slurm_job_files,
    create_slurm_jobarrays,
    create_slurm_single_job_file,
)

BASE_PROBLEMS = {
    "MNIST_LENET5": {
        "loss_func": "logloss",
        "dataset": "mnist",
        "model": "lenet5",
    },
    "CIFAR10_RESNET18": {
        "loss_func": "logloss",
        "dataset": "cifar10",
        "model": "resnet18",
    },
    "CIFAR10_RESNET34": {
        "loss_func": "logloss",
        "dataset": "cifar10",
        "model": "resnet34",
    },
    "CIFAR10_RESNET50": {
        "loss_func": "logloss",
        "dataset": "cifar10",
        "model": "resnet50",
    },
    "CIFAR100_RESNET50": {
        "loss_func": "logloss",
        "dataset": "cifar100",
        "model": "resnet50",
    },
    "CIFAR100_RESNET34": {
        "loss_func": "logloss",
        "dataset": "cifar100",
        "model": "resnet34",
    },
    "PTB_TRANSFORMERXL": {
        "loss_func": "logloss",
        "dataset": "ptb",
        "model": "transformer_xl",
        "model_args": {
            "n_layer": 6,
            "d_model": 512,
            "n_head": 8,
            "d_head": 64,
            "d_inner": 2048,
            "dropout": 0.1,
            "dropatt": 0.0,
            "tgt_len": 128,
            "mem_len": 128,
        },
    },
    "PTB_TRANSFORMERXL_DET": {
        "loss_func": "logloss",
        "dataset": "ptb",
        "model": "transformer_xl_deterministic",
        "model_args": {
            "n_layer": 6,
            "d_model": 512,
            "n_head": 8,
            "d_head": 64,
            "d_inner": 2048,
            "tgt_len": 128,
            "mem_len": 128,
        },
    },
    "WT2_TRANSFORMERXL": {
        "loss_func": "logloss",
        "dataset": "wikitext2",
        "model": "transformer_xl",
        "model_args": {
            "n_layer": 6,
            "d_model": 512,
            "n_head": 8,
            "d_head": 64,
            "d_inner": 2048,
            "dropout": 0.1,
            "dropatt": 0.0,
            "tgt_len": 128,
            "mem_len": 128,
        },
    },
    "WT2_TRANSFORMERXL_DET": {
        "loss_func": "logloss",
        "dataset": "wikitext2",
        "model": "transformer_xl_deterministic",
        "model_args": {
            "n_layer": 6,
            "d_model": 512,
            "n_head": 8,
            "d_head": 64,
            "d_inner": 2048,
            "tgt_len": 128,
            "mem_len": 128,
        },
    },
    "WT2_TENC": {
        "loss_func": "logloss",
        "dataset": "wikitext2",
        "model": "transformer_encoder",
        "model_args": {
            "tgt_len": 35,
        },
    },
    "PTB_TENC": {
        "loss_func": "logloss",
        "dataset": "ptb",
        "model": "transformer_encoder",
        "model_args": {
            "tgt_len": 35,
        },
    },
    "WT2_TENC_DET": {
        "loss_func": "logloss",
        "dataset": "wikitext2",
        "model": "transformer_encoder_deterministic",
        "model_args": {
            "tgt_len": 35,
        },
    },
    "PTB_TENC_DET": {
        "loss_func": "logloss",
        "dataset": "ptb",
        "model": "transformer_encoder_deterministic",
        "model_args": {
            "tgt_len": 35,
        },
    },
    "DB_SQD": {
        "dataset": "squad",
        "model": "distilbert_base_pretrained",
        "model_args": {
            "tgt_len": 384,
            "doc_stride": 128,
        },
    },
}

PROB_MNIST_LENET5 = BASE_PROBLEMS["MNIST_LENET5"]
PROB_CIFAR10_RESNET18 = BASE_PROBLEMS["CIFAR10_RESNET18"]
PROB_CIFAR10_RESNET34 = BASE_PROBLEMS["CIFAR10_RESNET34"]
PROB_CIFAR10_RESNET50 = BASE_PROBLEMS["CIFAR10_RESNET50"]
PROB_CIFAR100_RESNET50 = BASE_PROBLEMS["CIFAR100_RESNET50"]
PROB_CIFAR100_RESNET34 = BASE_PROBLEMS["CIFAR100_RESNET34"]
PROB_PTB_TRANSFORMERXL = BASE_PROBLEMS["PTB_TRANSFORMERXL"]
PROB_WT2_TXL = BASE_PROBLEMS["WT2_TRANSFORMERXL"]
PROB_WT2_TXL_DET = BASE_PROBLEMS["WT2_TRANSFORMERXL_DET"]
PROB_WT2_TENC = BASE_PROBLEMS["WT2_TENC"]
PROB_PTB_TENC = BASE_PROBLEMS["PTB_TENC"]
PROB_PTB_TENC_DET = BASE_PROBLEMS["PTB_TENC_DET"]
PROB_DB_SQD = BASE_PROBLEMS["DB_SQD"]


def make_exp_dict_list_unique(experiments):
    experiments_str = [exp_dict_to_str(exp_dict) for exp_dict in experiments]
    experiments_str = list(set(experiments_str))
    return [exp_dict_from_str(exp_str) for exp_str in experiments_str]


def nice_logspace(start, end, base, density):
    """Returns a log-spaced grid between ``base**start`` and ``base**end``.

    Plays nicely with ``merge_grids``. Increasing the density repeats previous values.

    - ``Start``, ``end`` and ``density`` are integers.
    - Increasing ``density`` by 1 doubles the number of points.
    - ``Density = 1`` will return ``end - start + 1`` points
    - ``Density = 2`` will return ``2*(end-start) + 1`` points
    - ``nice_logspace(0, 1, base=10, density=1) == [1, 10] == [10**0, 10**1]``
    - ``nice_logspace(0, 1, base=10, density=2) == [1, 3.16..., 10] == [10**0, 10**(1/2), 10**1]``
    """
    if density < 0 or not np.allclose(int(density), density):
        raise ValueError(
            f"nice_logspace: density needs to be an integer >= 0, got {density}."
        )
    if not np.allclose(int(start), start) or not np.allclose(int(end), end):
        raise ValueError(
            f"nice_logspace: start and end need to be integers, got {start, end}."
        )
    assert end > start
    return np.logspace(start, end, base=base, num=(end - start) * (2**density) + 1)


def merge_grids(*grids):
    """Merge two lists of parameters.

    Given lists [a,b,c], [c,d,e], returns [a,b,c,d,e]
    """
    return sorted(list(set.union(*[set(grid) for grid in grids])))


def dict_update(x, y):
    """Non-mutable version of x.update(y)"""
    z = x.copy()
    z.update(y)
    return z


def merge_dicts(*dicts):
    """Merge dictionary, preserves keys from the right-most dicts. Equivalent
    to chaining ``x.update(y)``

    Example:
    .. code:: python

        merge_dicts({"a":1, "b":2}, {"b":3, "c":4}) == {"a":1, "b":3, "c":4}
    """
    z = dicts[0].copy()
    for other_dict in dicts[1:]:
        z.update(other_dict)
    return z


def merge_sets(*many_lists):
    """Merge lists without duplicates."""
    z = [x for x in many_lists[0]]
    for other_list in many_lists[1:]:
        z += [x for x in other_list]
    return list(set(z))


def experiment_maker_cli(
    exp_name,
    descr,
    experiments,
    hyperparam_names=None,
    as_one_job=False,
    as_job_array=False,
):
    """Creates the experiment json files necessary to send individual jobs.

    Will output the necessary files in

    .. code:: bash

        explib_workspace/
        └─ exp_name/
           ├─ exp_defs/
           │  ├─ ...
           │  ├─ exp_name_uuid1.json
           │  └─ exp_name_uuid2.json
           ├─ jobs/
           │  ├─ main.sh
           │  ├─ job_uuid1.sh
           │  └─ job_uuid2.sh
           └─ exp_name_summary.json

    ``exp_name`` needs to be filename friendly (to be safe, use ``[a-z]|[A-Z]|-|_``)

    - If ``as_one_job``, outputs only one sbatch file to runs all jobs sequentially
    - If ``as_job_array``, outputs one sbatch file to submit every job for slurm config in parallel
    - By default, creates one sbatch file per experiment

    ``hyperparam_names`` is a list of hyperparameter names in wandb format
    (eg ``"opt.name"``) for a better ``--report`` option.

    Calling this file with

    - No arguments: Generates the exp_defs and jobs as above.

    - ``--report``: Queries wandb and prints a summary of which jobs have finished
      (and have been checked by ``python -m explib.results checkup``)

      If no ``hyperparam_names`` are given,
      ``--report`` outputs only the percentage of finished runs on wandb.
      If ``hyperparam_names`` is given, gives a breakdown per hyper-parameter value.

    - ``--unfinished``: Queries wandb; only generates jobs for unfinished runs
      as logged on wandb (the missing runs from ``--report``)
    """
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generates a report on what experiments have been run/are stored on wandb",
        default=False,
    )
    parser.add_argument(
        "--report-short",
        action="store_true",
        help="Short reporting. Does not download summary, need to call python -m explib.results --summary first.",
        default=False,
    )
    parser.add_argument(
        "--unfinished",
        action="store_true",
        help="Only generate sbatch scripts for experiments that are not finished on wandb.",
        default=False,
    )
    cli_helper.add_dotenv_option(parser)
    args = parser.parse_args()
    cli_helper.load_dotenv_if_required(args)

    loglevel = (
        "WARNING"
        if args.report or args.report_short
        else config.get_console_logging_level()
    )

    logging.init_logging_stdout(level=loglevel)
    logging.info(f"Creating files for {exp_name} in {get_exp_def_folder(exp_name)}")
    logging.info(f"Experiment description: {descr}")
    create_experiment_definitions(exp_name, experiments)

    if args.report:
        wandb_reporting.check_status(exp_name, hyperparam_names)
    elif args.report_short:
        hyperparam_names = ["seed", "slurm_config"]
        wandb_reporting.check_status(exp_name, hyperparam_names, download=False)
    else:
        filter_should_run_wuuid = lambda wuuid: True
        if args.unfinished:
            logging.info("Creating filter to not re-run already run experiments")
            uuid_has_run = wandb_reporting.wuuid_to_successful_run(exp_name)
            filter_should_run_wuuid = lambda wuuid: not uuid_has_run[wuuid]

        if as_one_job:
            create_slurm_single_job_file(exp_name, filter_should_run_wuuid)
        elif as_job_array:
            create_slurm_jobarrays(exp_name, filter_should_run_wuuid)
        else:
            create_slurm_job_files(exp_name, filter_should_run_wuuid)
