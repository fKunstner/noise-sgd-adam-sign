import json
import re
from datetime import datetime, timedelta
import explib.results.data as data_h
import numpy as np
import warnings
import pandas as pd

from explib import logging


def clean_data(summary, runs):
    """
    All the data cleanup such that the summary and run data can be plotted.

    Expects summary and runs to be dataframes,
    returns cleaned dataframes.
    """

    #
    summary, runs = convert_runtimes_to_seconds(summary, runs)
    summary, runs = fill_runtime_from_missing_summary(summary, runs)
    #
    summary, runs = process_tags(summary, runs)
    summary, runs = drop_bad_tags(summary, runs)
    summary, runs = remove_crashes(summary, runs)
    summary, runs = remove_duplicates(summary, runs)
    #
    summary, runs = bump_epoch_counter(summary, runs)
    #
    summary, runs = fill_accumulate_steps(summary, runs)
    summary, runs = add_dataset_size(summary, runs)
    summary, runs = rescale_training_loss_for_squad_with_accumulation(summary, runs)
    #
    summary, runs = add_update_count(summary, runs)
    summary, runs = rescale_accuracy(summary, runs)
    #
    summary, runs = fill_in_defaults(summary, runs)

    return summary, runs


def fill_accumulate_steps(summary, runs):
    """If accumulate steps wasn't set, uses 1 by default"""

    summary["accumulate_steps"] = np.where(
        summary["accumulate_steps"].isna(), 1, summary["accumulate_steps"]
    )
    return summary, runs


def add_dataset_size(summary, runs):
    """Add in dataset size and numbers related to it;
    the number of minibatches per epoch
    and the number of gradient updates per epoch
    """
    ds_to_size = {
        "mnist": 60000,
        "cifar10": 50000,
        "ptb": 26560,
        "wikitext2": 16317,
        "squad": 87714,
    }

    summary["ds_size"] = np.nan
    for k in ds_to_size.keys():
        summary["ds_size"] = np.where(
            summary["dataset"] == k, ds_to_size[k], summary["ds_size"]
        )

    summary["grad_updates_per_epoch"] = np.floor(
        summary["ds_size"] / (summary["batch_size"] * summary["accumulate_steps"])
    )

    # max(summary["grad_updates_per_epoch"], 1)
    summary["grad_updates_per_epoch"] = np.where(
        summary["grad_updates_per_epoch"] == 0, 1, summary["grad_updates_per_epoch"]
    )

    summary["minibatches_per_epoch"] = np.floor(
        summary["ds_size"] / (summary["batch_size"])
    )

    runs = pd.merge(
        left=runs,
        right=summary[["dataset", "grad_updates_per_epoch", "minibatches_per_epoch"]],
        left_on="id",
        right_index=True,
        how="left",
    )

    return summary, runs


def add_update_count(summary, runs):
    """Adds a column counting the number of parameter updates rather than epochs"""
    runs["update_count"] = runs["epoch"] * runs["grad_updates_per_epoch"]
    return summary, runs


def rescale_accuracy(summary, runs):
    summary["train_accuracy"] = 100 * summary["train_accuracy"]
    summary["valid_accuracy"] = 100 * summary["valid_accuracy"]
    runs["train_accuracy"] = 100 * runs["train_accuracy"]
    runs["valid_accuracy"] = 100 * runs["valid_accuracy"]
    return summary, runs


def rescale_training_loss_for_squad_with_accumulation(summary, runs):
    """
    In experiment code, we messed up the computation of the loss
    by a constant factor due to the use of gradient accumulation.
    This fixes it.

    Instead of dividing by the number of gradient computation
    to obtain the average (of the average loss over datapoints) over batches,
    we divided by the number of effective gradient steps.

    This leads to an obvious issue where multiplying the number of accumulation
    steps by 4 would lead to a 4-fold increase in the loss.

    To fix this, we can multiply by the number of effective steps per epoch
    and divide by the number of accumulation steps.

    They're off by a factor
    """
    summary["training_loss"] = np.where(
        summary["dataset"] == "squad",
        summary["training_loss"]
        * summary["grad_updates_per_epoch"]
        / summary["minibatches_per_epoch"],
        summary["training_loss"],
    )

    runs["training_loss"] = np.where(
        runs["dataset"] == "squad",
        runs["training_loss"]
        * runs["grad_updates_per_epoch"]
        / runs["minibatches_per_epoch"],
        runs["training_loss"],
    )
    return summary, runs


def bump_epoch_counter(summary, runs):
    runs["epoch"] = runs["epoch"] + 1
    runs["epoch"] = np.where(runs["step"] == 0, 0, runs["epoch"])
    return summary, runs


def remove_crashes(summary, runs):
    summary = data_h.df_unselect(summary, status="OOM")
    summary = data_h.df_unselect(summary, status="Crash")
    summary = data_h.df_unselect(summary, status="NoLogs")
    return summary, runs


def process_tags(summary, runs):
    summary["tags"] = summary["tags"].apply(lambda x: json.loads(x.replace("'", '"')))
    summary["bad_tag"] = summary["tags"].apply(
        lambda tags: any([tag.startswith("bad-") for tag in tags])
    )
    return summary, runs


def drop_bad_tags(summary, runs):
    summary = summary.loc[~summary["bad_tag"]]
    return summary, runs


def remove_duplicates(summary, runs):
    summary["is_duplicate"] = summary["tags"].apply(lambda x: "duplicate" in x)
    summary = data_h.df_select(summary, is_duplicate=False)
    summary = summary.drop_duplicates(subset="wuuid", keep="last")
    return summary, runs


def convert_runtimes_to_seconds(summary, runs):
    def exp_runtime_to_seconds(timestring):
        if type(timestring) != str:  # is most likely nan
            return timestring
        elif timestring == "":
            return np.nan

        hours_correction = 0
        if "days" in timestring:
            match = re.search("([0-9])\sdays,\s(.*)", timestring)
            n_days = int(match.group(1))
            hours_correction = n_days * 24
            time_bit = match.group(2)
            t = datetime.strptime(time_bit, "%H:%M:%S.%f")
        elif "day" in timestring:
            match = re.search("(1)\sday,\s(.*)", timestring)
            hours_correction = 24
            time_bit = match.group(2)
            t = datetime.strptime(time_bit, "%H:%M:%S.%f")
        else:
            t = datetime.strptime(timestring, "%H:%M:%S.%f")

        delta = timedelta(
            hours=t.hour + hours_correction, minutes=t.minute, seconds=t.second
        )
        return delta.total_seconds()

    try:
        summary["exp_runtime_s"] = summary["exp_runtime"].apply(exp_runtime_to_seconds)
        runs["exp_runtime_s"] = runs["exp_runtime"].apply(exp_runtime_to_seconds)
    except:
        import pdb

        pdb.set_trace()

    return summary, runs


def fill_runtime_from_missing_summary(summary, runs):
    """Fill missing values in summary.

    Currently only fills in the exp_runtime
    """
    missing_runtime = summary[summary["exp_runtime_s"].isna()]
    missing_summary = summary[summary["status"] == "Finished with no summary"]
    if missing_runtime.shape[0] != missing_summary.shape[0]:
        logging.debug(
            (
                f"The number of runs with missing summary ({missing_summary.shape[0]}) "
                + f"does not match the number of runs with missing runtime ({missing_runtime.shape[0]}). "
                + "This might indicate data issues on wandb."
            )
        )

    runs_runtime = runs.groupby("id")["exp_runtime_s"].max()
    merged = pd.merge(left=summary, right=runs_runtime, on="id", how="left")
    merged = merged.drop(columns=["exp_runtime_s_x", "exp_runtime"])
    merged = merged.rename(columns={"exp_runtime_s_y": "exp_runtime"})

    n_missing = merged[merged["exp_runtime"].isna()].shape[0]
    if n_missing > 0:
        logging.debug(
            (
                "Filling of runtime failed (?). "
                + f"There are still {n_missing} missing values."
            )
        )

    return summary, runs


def fill_in_defaults(summary, runs):
    summary = summary.fillna(value={"opt.momentum": 0.0})
    return summary, runs


def sanity_check_number_of_runs(bs, dataset, max_epoch, model, runs_df):
    if any(runs_df.groupby("opt.alpha").count()["id"] < 3):
        print(
            f"Issue with {model} {dataset} {max_epoch} {bs}: Too few runs for one of the step-sizes!"
        )
    if any(runs_df.groupby("opt.alpha").count()["id"] > 3):
        print(
            f"Issue with {model} {dataset} {max_epoch} {bs}: Too many runs for one of the step-sizes!"
        )
        print(runs_df.groupby("opt.alpha").count())


def filter_merge(summary, runs, summary_filter, runs_filter):
    summary_filtered = data_h.df_select(summary, **summary_filter)
    runs_filtered = data_h.df_select(runs, **runs_filter)

    runs_filtered = runs_filtered.loc[runs_filtered["id"].isin(summary_filtered.index)]

    merged = runs_filtered.join(
        other=summary_filtered,
        on="id",
        how="right",
        lsuffix="_runs",
        rsuffix="_summary",
    )

    return merged
