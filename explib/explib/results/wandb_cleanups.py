import pdb

from explib import config
from explib.expmaker.experiment_defs import (
    exp_dict_from_str,
    exp_dict_to_str,
    make_uuid,
    make_wuuid,
)
from explib.results.data import WandbAPI
from tqdm import tqdm

##
# Helper functions


def get_logs(run):
    """Downloads the .log file for the run and returns its content as a str."""
    files = run.files()

    for file in files:
        if file.name == "output.log":
            fh = file.download(replace=True)
            logs = fh.read()

            return logs
    return False


def get_exp_dict_str(arun):
    """Extract the string representation of exp_dict from the run's logs."""
    logs = get_logs(arun)
    if logs == False:
        return False

    firstline = logs.split("\n")[0]

    if firstline.startswith("Downloading"):
        firstline = logs.split("\n")[2]

    dictionary_parts = firstline.split("dictionnary ")
    if len(dictionary_parts) != 2:
        print(f"Error; {len(dictionary_parts)} dictionary parts instead of 2.")
        print("Don't know what to do. Here's a debugger:")
        pdb.set_trace()
    dictionary_part = dictionary_parts[1]
    exp_dict_str = dictionary_part.replace("'", '"')
    return exp_dict_str


##
# Updating the wuuid and expconfig_str for experiments that ran on older
# versions of explib


def update_wuuid(arun):
    """Migrates to the new WUUID system.

    For runs that do not have a WUUID, creates it and updates the run.
    Also adds in the field expconfig_str.

    Context:
    We used to have only one UUID per experiments, now we have two (UUID and WUUID).
    The UUID depends on the entire exp_dict, including slurm_config,
    and can not be used to check for finished jobs across clusters with different
    slurm configs. The WUUID is the same as UUID but independent of exp_dict.
    """
    if "wuuid" in arun.config:
        return

    exp_dict_str = get_exp_dict_str(arun)
    if exp_dict_str == False:
        return

    exp_dict = exp_dict_from_str(exp_dict_str)
    predicted_uuid = make_uuid(exp_dict)

    if arun.config["uuid"] != predicted_uuid:
        print("Error: predicted uuid doesn't match stored uuid.")
        print("Don't know what to do. Here's a debugger:")
        pdb.set_trace()

    arun.config["wuuid"] = make_wuuid(exp_dict)
    arun.config["exp_dict_str"] = exp_dict_to_str(exp_dict)
    print(f"UPDATE_WUUID: Updated wuuid and exp_dict_str for {arun.id}")
    arun.update()


##
# Flag run status
# Rules and how to apply them


def _rule_success(run):
    if "max_epoch" not in run.config:
        return False
    if "epoch" not in run.summary:
        return False
    return run.config["max_epoch"] == run.summary["epoch"] + 1


def _rule_diverged(run):
    return "training_error" in run.summary


def _rule_diverged_from_logs(run):
    logs = get_logs(run)
    if logs:
        if "{'training_error': 'nan'}" in logs:
            return True
    return False


def _rule_OOM(run):
    logs = get_logs(run)
    if logs:
        if "CUDA error: out of memory" in logs:
            return True
    return False


def _rule_finished_with_no_summary(run):
    if dict(run.summary) == {} or dict(run.summary) == {
        "status": "Finished with no summary"
    }:
        if len(run.history(samples=1, keys=["exp_runtime"])) == 1:
            return True
    return False


def _rule_uncaught_exception(run):
    logs = get_logs(run)
    if logs == False:
        return False
    if "Uncaught exception" in logs:
        return True
    return False


def _rule_no_logs(run):
    if get_logs(run) == False:
        return True
    return False


_flags_and_rules = [
    ("Success", _rule_success),
    ("Finished with no summary", _rule_finished_with_no_summary),
    ("Diverged", _rule_diverged),
    ("NoLogs", _rule_no_logs),
    ("OOM", _rule_OOM),
    ("Crash", _rule_uncaught_exception),
    ("Diverged", _rule_diverged_from_logs),
]


def flag_status(arun):
    if "status" in arun.summary:
        return

    status_found = False
    for (flag, rule) in _flags_and_rules:
        if rule(arun):
            arun.summary["status"] = flag
            print(f"FLAG_STATUS: Marking run {arun.id} as {flag}")
            arun.update()
            status_found = True
            break

    if not status_found:
        print(
            "FLAG_STATUS: Unknown run status! Don't know what to do. Here's a debugger:"
        )
        pdb.set_trace()


##
# Unify expdict_str and exp_dict_str


def expdict_str_rename(arun):
    if "expdict_str" in arun.config:
        arun.config["exp_dict_str"] = arun.config["expdict_str"]
        arun.config.pop("expdict_str", None)
        print(f"EXPDICT_STR_RENAME: Renamed expdict_str to exp_dict_str for {arun.id}")
        arun.update()


##
#

##
#


def correct_experiment_definition():
    """Fix an issue in some experiment definitions.

    Problem:
        The full batch experiments were intended to run in full batch
        (using ``drop_last = True, shuffle = False``, effectively dropping the last incomplete batch)
        but did not, because ``shuffle`` was never passed to the dataset ``__init__``.

        - The experiments using ``MNIST`` and ``CIFAR10`` were not affected. There was no incomplete batch to drop, ``shuffle`` did not affect what gets dropped.
        - It did not affect ``PTB`` and ``WikiText2`` because ``shuffle`` was not implemented
          The code from the Transformer-XL paper used an ordered iterator rather than
          a shuffled one
        - The only affected dataset for the full batch experiments were on ``SQuAD``.

    The run are still valid, but the ``shuffle=True`` should be turned to ``False``.

    Solution:
        This script updates runs from wandb as follows:

        If the run has ``shuffle = False``,
        remove the ``shuffle`` key and update the exp dict and the unique ids.

    Args:
        arun: The wandb run to potentially fix
    """
    runs = WandbAPI.get_handler().runs(config.get_wandb_project(), per_page=1000)
    for arun in tqdm(runs):
        if "shuffle" in arun.config and arun.config["shuffle"] == False:
            exp_dict = exp_dict_from_str(arun.config["exp_dict_str"])
            predicted_uuid = make_uuid(exp_dict)
            predicted_wuuid = make_wuuid(exp_dict)

            if arun.config["shuffle"] != exp_dict["shuffle"]:
                pdb.set_trace(
                    header="Error: exp_dict and wandb shuffle don't match. Don't know what to do, here's a debugger:"
                )
            if arun.config["uuid"] != predicted_uuid:
                pdb.set_trace(
                    header="Error: predicted and stored uuid don't match. Don't know what to do, here's a debugger:"
                )
            if arun.config["wuuid"] != predicted_wuuid:
                pdb.set_trace(
                    header="Error: predicted and stored wuuid don't match. Don't know what to do, here's a debugger:"
                )

            exp_dict.pop("shuffle")
            arun.config.pop("shuffle")

            arun.config["exp_dict_str"] = exp_dict_to_str(exp_dict)
            arun.config["wuuid"] = make_wuuid(exp_dict)
            arun.config["uuid"] = make_uuid(exp_dict)

            arun.tags.append("RemovedFalseShuffle")

            print(f"FIX_SHUFFLE: RemovedFalseShuffle for {arun.id}")
            arun.update()


def checkup(group=None):
    filters = {"group": group} if group is not None else {}
    runs = WandbAPI.get_handler().runs(
        config.get_wandb_project(), filters=filters, per_page=1000
    )
    for arun in tqdm(runs):
        flag_status(arun)
        update_wuuid(arun)
        expdict_str_rename(arun)
