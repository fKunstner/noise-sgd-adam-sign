import os
import textwrap

from explib import config, logging
from explib.expmaker import (
    slurm_configs,
    get_jobs_folder,
    get_exp_full_path_json,
    load_summary,
    get_job_path,
)
from explib.expmaker.experiment_defs import make_uuid, make_wuuid
from explib.expmaker.slurm_configs import SlurmConfigIssue


def make_sbatch_config(slurm_config_name, jobarray_len=None):
    slurm_config = slurm_configs.SLURM_CONFIGS[slurm_config_name]
    gpu_str = ""
    if slurm_config["gpu"] is not None:
        gpu_str = "#SBATCH --gres=gpu:{gpu}".format(gpu=slurm_config["gpu"])
    notification_str = ""
    if config.get_notification_email() is not None:
        notification_str = textwrap.dedent(
            f"""
            #SBATCH --mail-user={config.get_notification_email()}
            #SBATCH --mail-type=ALL
            """
        )
    jobarray_str = ""
    if jobarray_len is not None:
        jobarray_str = textwrap.dedent(
            f"""
            #SBATCH --array=0-{jobarray_len-1}
            """
        )
    account_or_partition_str = f"#SBATCH --account={config.get_slurm_account()}"
    if config.get_slurm_partition() is not None:
        account_or_partition_str = f"#SBATCH --partition={config.get_slurm_partition()}"

    conda_load_env_str = ""
    if config.get_conda_env() is not None:
        conda_load_env_str = f"conda activate {config.get_conda_env()}"
    env_load_str = ""
    if config.get_env_file_to_source() is not None:
        env_load_str = f". {config.get_env_file_to_source()}"

    return textwrap.dedent(
        """
        {account_or_partition_str}
        #SBATCH --mem={mem}
        #SBATCH --time={time}
        #SBATCH --cpus-per-task={cpus}
        {notification_str}
        {gpu_str}
        {jobarray_str}
        
        {conda_load_env_str}
        {env_load_str}
        export TMPDIR=$SLURM_TMPDIR
        """
    ).format(
        account_or_partition_str=account_or_partition_str,
        mem=slurm_config["mem"],
        time=slurm_config["time"],
        cpus=slurm_config["cpus-per-task"],
        notification_str=notification_str,
        gpu_str=gpu_str,
        jobarray_str=jobarray_str,
        conda_load_env_str=conda_load_env_str,
        env_load_str=env_load_str,
    )


def filter_experiments_for_slurm_config(summary, slurm_config_name):
    """Filters an experiment dictionary summary for experiments with slurm_config.

    `experiment_dicts` is a dictionary of `hash => exp_dict`

    Raises a SlurmConfigIssue if there is an experiment with no slurm config
    """
    filtered_dict = {}
    for key, exp_dict in summary.items():
        if "slurm_config" not in exp_dict:
            raise SlurmConfigIssue(
                f"Slurm config not found in experiment {key}. Full exp def: {exp_dict}."
            )
        if exp_dict["slurm_config"] == slurm_config_name:
            filtered_dict[key] = exp_dict
    return filtered_dict


def make_slurm_job(exp_dict, json_path, out_path):
    if "slurm_config" not in exp_dict:
        raise SlurmConfigIssue(f"No slurm_config in experiment dictionary ({exp_dict})")

    config_name = exp_dict["slurm_config"]
    if config_name not in slurm_configs.SLURM_CONFIGS:
        raise SlurmConfigIssue(
            f"Slurm config '{config_name}' unknown"
            + f"(possible: {slurm_configs.SLURM_CONFIGS.keys()})"
        )

    with open(out_path, "w+") as file:
        file.writelines(
            textwrap.dedent(
                """#!/bin/sh
                
                {sbatch_config}
                
                python -m explib {json_path}
                
                exit
                
                """
            ).format(
                sbatch_config=make_sbatch_config(config_name),
                json_path=json_path,
            )
        )
        file.close()


def make_jobarray_file(exp_name, summary, slurm_config_name, filter_should_run_wuuid):
    jobs_folder = get_jobs_folder(exp_name)
    out_path = os.path.join(
        jobs_folder, f"run_all_{exp_name}_array_{slurm_config_name}.sh"
    )
    summary_for_config = filter_experiments_for_slurm_config(summary, slurm_config_name)

    summary_for_config_filtered = {
        make_wuuid(exp_dict): exp_dict
        for uuid, exp_dict in summary_for_config.items()
        if filter_should_run_wuuid(make_wuuid(exp_dict))
    }

    with open(out_path, "w+") as file:
        file.writelines(
            textwrap.dedent(
                """#!/bin/sh
                
                {sbatch_config}
                
                """
            ).format(
                sbatch_config=make_sbatch_config(
                    slurm_config_name, jobarray_len=len(summary_for_config_filtered)
                ),
            )
        )
        file.writelines(
            [
                textwrap.dedent(
                    f"""
                    if [ $SLURM_ARRAY_TASK_ID -eq {i} ]
                    then 
                        python -m explib {get_exp_full_path_json(exp_name, exp_dict)}
                    fi
                    
                    """
                )
                for i, exp_dict in enumerate(summary_for_config_filtered.values())
            ]
        )
        file.writelines("exit")
        file.writelines("\n")
        file.close()
    logging.info(f"Created job array file for config {slurm_config_name} at {out_path}")


def create_slurm_jobarrays(exp_name, filter_should_run_wuuid=None):
    """Creates one jobarray file per SLURM config"""
    summary = load_summary(exp_name)

    if filter_should_run_wuuid is None:
        filter_should_run_wuuid = lambda x: True

    unique_slurm_configs = list(
        set([exp_dict.get("slurm_config", None) for exp_dict in summary.values()])
    )
    for slurm_config in unique_slurm_configs:
        make_jobarray_file(exp_name, summary, slurm_config, filter_should_run_wuuid)


def create_slurm_single_job_file(exp_name, filter_should_run_wuuid=None):
    """Creates one sbatch file to run all experiments sequentially"""
    summary = load_summary(exp_name)

    if filter_should_run_wuuid is None:
        filter_should_run_wuuid = lambda x: True

    logging.info(f"Checking json files...")
    unique_slurm_configs = set(
        [exp_dict.get("slurm_config", None) for exp_dict in summary.values()]
    )
    if len(unique_slurm_configs) != 1:
        raise SlurmConfigIssue(
            f"Expected single Slurm config, multiple requested {unique_slurm_configs}"
        )
    slurm_config_name = unique_slurm_configs.pop()

    jobs_folder = get_jobs_folder(exp_name)
    out_path = os.path.join(jobs_folder, f"run_all_{exp_name}.sh")
    with open(out_path, "w+") as file:
        file.writelines(
            textwrap.dedent(
                """#!/bin/sh
                
                {sbatch_config}
                
                """
            ).format(
                sbatch_config=make_sbatch_config(slurm_config_name),
            )
        )
        file.writelines(
            [
                textwrap.dedent(
                    f"""
                    python -m explib {get_exp_full_path_json(exp_name, exp_dict)}
                    """
                )
                for exp_dict in summary.values()
                if filter_should_run_wuuid(make_wuuid(exp_dict))
            ]
        )
        file.writelines("exit")
        file.writelines("\n")
        file.close()
    logging.info(f"Created single job file, run with > sbatch {out_path}")


def create_slurm_job_files(exp_name, filter_should_run_wuuid=None):
    """Creates one sbatch file per experiment"""
    jobs_folder = get_jobs_folder(exp_name)

    if filter_should_run_wuuid is None:
        filter_should_run_wuuid = lambda x: True

    summary = load_summary(exp_name)

    try:
        logging.info(f"Creating job files in {jobs_folder}")
        out_paths = []
        for uuid, exp_dict in summary.items():
            if filter_should_run_wuuid(make_wuuid(exp_dict)):
                json_path = get_exp_full_path_json(exp_name, exp_dict)
                out_path = get_job_path(exp_name, exp_dict)
                make_slurm_job(exp_dict, json_path, out_path)
                out_paths.append(out_path)
        logging.info(f"Created {len(out_paths)} job files")

        run_all_filepath = os.path.join(get_jobs_folder(exp_name), f"run_{exp_name}.sh")
        with open(run_all_filepath, "w+") as fp:
            fp.writelines([f"sbatch {out_path}\n" for out_path in out_paths])
        logging.info(f"Submit all jobs with > source {run_all_filepath}")

    except SlurmConfigIssue as e:
        logging.warn("Slurm config not found - skipping making sbatch files")
        logging.warn(e, exc_info=1)
