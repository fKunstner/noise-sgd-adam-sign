import base64
import hashlib
import json
import os
from pathlib import Path
from explib import config, logging


def exp_dict_from_str(exp_dict_str):
    return json.loads(exp_dict_str)


def exp_dict_to_str(exp_dict, remove_keys=None):
    """String version of the experiment dictionary"""
    if remove_keys is not None:
        exp_dict_copy = json.loads(json.dumps(exp_dict))
        for key in remove_keys:
            if key in exp_dict_copy:
                del exp_dict_copy[key]
        return json.dumps(exp_dict_copy, sort_keys=True)
    return json.dumps(exp_dict, sort_keys=True)


def _uuid_from_str(string):
    bytes = string.encode("ascii")
    hash = hashlib.sha256(bytes)
    # Using base 32 as it is filesystem-safe for Unix and Windows
    return base64.b32encode(hash.digest()).decode("ascii")


def make_wuuid(exp_dict):
    """Wandb UUID (ignores slurm_config for multi-cluster)"""
    return _uuid_from_str(exp_dict_to_str(exp_dict, remove_keys=["slurm_config"]))


def make_uuid(exp_dict):
    """Wandb UUID derived from exp_dict"""
    return _uuid_from_str(exp_dict_to_str(exp_dict))


def gen_uuid_to_exp_dicts(experiments):
    """Get a dictionary of uuids => experiment dictionary"""
    return {make_uuid(exp_dict): exp_dict for exp_dict in experiments}


def get_exp_folder(exp_name):
    folder = os.path.join(config.get_workspace(), exp_name)
    Path(folder).mkdir(parents=True, exist_ok=True)
    return folder


def get_exp_def_folder(exp_name):
    folder = os.path.join(get_exp_folder(exp_name), "exp_defs")
    Path(folder).mkdir(parents=True, exist_ok=True)
    return folder


def get_jobs_folder(exp_name):
    jobs_folder = os.path.join(get_exp_folder(exp_name), "jobs")
    Path(jobs_folder).mkdir(parents=True, exist_ok=True)
    return jobs_folder


def get_exp_full_path_json(exp_name, exp_dict):
    uuid = make_uuid(exp_dict)
    filename = f"{exp_name}_{uuid}.json"
    return os.path.join(get_exp_def_folder(exp_name), filename)


def get_job_path(exp_name, exp_dict):
    uuid = make_uuid(exp_dict)
    filename = f"{exp_name}_{uuid}.sh"
    return os.path.join(get_jobs_folder(exp_name), filename)


def create_experiment_definitions(exp_name, experiments):
    experiment_folder = get_exp_def_folder(exp_name)

    logging.info(f"Storing experiment files in {experiment_folder}")
    uuid_to_expdicts = gen_uuid_to_exp_dicts(experiments)
    for uuid, exp_dict in uuid_to_expdicts.items():
        exp_filepath = get_exp_full_path_json(exp_name, exp_dict)
        with open(exp_filepath, "w") as fp:
            json.dump(exp_dict, fp)
        logging.debug(f"Created {exp_filepath}")

    summary_filepath = os.path.join(
        get_exp_folder(exp_name), f"{exp_name}_summary.json"
    )
    with open(summary_filepath, "w") as fp:
        json.dump(uuid_to_expdicts, fp, indent=4)
    logging.info(f"Created {len(uuid_to_expdicts)} experiment files")
    logging.info(f"Summary in {summary_filepath}")


def load_summary(exp_name):
    summary_filepath = os.path.join(
        get_exp_folder(exp_name), f"{exp_name}_summary.json"
    )
    with open(summary_filepath, "r") as fp:
        summary = json.load(fp)
    return summary
