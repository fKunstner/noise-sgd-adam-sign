import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import wandb
from explib import config
from explib.results import experiment_groups as expdef
from tqdm import tqdm


class WandbAPI:
    """Static class to provide a singleton handler to the wandb api."""

    api_handler = None

    @staticmethod
    def get_handler():
        if WandbAPI.api_handler is None:
            WandbAPI.api_handler = wandb.Api()
        return WandbAPI.api_handler


##
#


def flatten_dict(x):
    return pd.io.json._normalize.nested_to_record(x)


##
# Folders and magic strings


def get_results_folder():
    path = os.path.join(config.get_workspace(), "results")
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_runs_folder():
    path = os.path.join(config.get_workspace(), "results", "runs")
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_run_path(id):
    return os.path.join(get_runs_folder(), f"{id}.csv")


SUMMARY_FILE = os.path.join(get_results_folder(), "summary.csv")
ALLRUNS_FILE = os.path.join(get_results_folder(), "all_runs.csv")


##
# Load data from disk, download if not exist


def list_converter(as_str):
    try:
        if len(as_str) == 0:
            return []
        as_str = as_str.replace("'Infinity'", "NaN")
        as_str = as_str.replace("'NaN'", "NaN")
        return json.loads(as_str)
    except:
        import pdb

        pdb.set_trace()


def get_summary(ignore_per_iter_data=False):
    """Returns a dataframe with summary information about all runs.

    Uses the last downloaded .csv if it exists, downloads a new one otherwise

    Can be updated manually with ``python -m explib.result --download-summary``
    or ``explib.results.data.download_summary``
    """
    if not os.path.isfile(SUMMARY_FILE):
        download_summary()

    def string_bool_table(default=True):
        DEFAULT = ""
        return {"True": True, "False": False, DEFAULT: default}

    mappings = {
        "shuffle": string_bool_table(default=True),
        "drop_last": string_bool_table(default=False),
        "opt.use_bias_correction": string_bool_table(default=True),
    }

    summary_converters = {
        **{k: lambda x: mappings[k][x] for k in mappings.keys()},
        "norm_squared_gradients": list_converter,
        "norm_squared_gradients_l1": list_converter,
        "function_values": list_converter,
    }

    extra_args = {}
    if ignore_per_iter_data:
        cols = list(pd.read_csv(SUMMARY_FILE, nrows=1))
        cols_to_drop = [
            "norm_squared_gradients",
            "norm_squared_gradients_l1",
            "function_values",
        ]
        usecols = [col for col in cols if col not in cols_to_drop]
        extra_args = {"usecols": usecols}

    return pd.read_csv(
        SUMMARY_FILE,
        header=0,
        index_col="id",
        converters=summary_converters,
        **extra_args,
    )


def get_run(id):
    """Returns a dataframe of all info for a run.

    Returns cached .csv if exists, otherwise downloads from wandb

    Force re-download with ``explib.results.data.download_run_by_id``
    """
    file_path = get_run_path(id)

    if not os.path.isfile(file_path):
        download_run_by_id(id)

    df = pd.read_csv(file_path, header=0, low_memory=False)

    return df


def get_all_runs(ignore_per_iter_data=False):
    """Returns a dataframe with info for all runs.

    Uses cached concatenated .csv file if it exists, create it otherwise

    Can be updated manually with ``python -m explib.result --concat-runs``
    or explib.results.data.concatenate_all_runs()
    """
    if not os.path.isfile(ALLRUNS_FILE):
        concatenate_all_runs()

    summary_converters = {
        "norm_squared_gradients": list_converter,
        "norm_squared_gradients_l1": list_converter,
        "function_values": list_converter,
    }

    extra_params = {}
    if ignore_per_iter_data:
        cols = list(pd.read_csv(ALLRUNS_FILE, nrows=1))
        cols_to_drop = [
            "norm_squared_gradients",
            "norm_squared_gradients_l1",
            "function_values",
        ]
        usecols = [col for col in cols if col not in cols_to_drop]
        extra_params = {"usecols": usecols}

    return pd.read_csv(
        ALLRUNS_FILE,
        header=0,
        converters=summary_converters,
        dtype={"exp_runtime": "str"},
        **extra_params,
    )


##
# Data download


def download_run_by_id(id):
    """See `download_run`"""
    run = WandbAPI.get_handler().run(config.get_wandb_project() + "/" + id)
    download_run(run)


def download_run(arun: wandb.apis.public.Run):
    """Given a Wandb Run, download the full history."""
    df = arun.history(samples=arun._attrs["historyLineCount"], pandas=(True))
    df.to_csv(get_run_path(arun.id))


def download_summary(download_runs=False, group=None, only_new=False):
    """Download a summary of all runs on the wandb project."""
    filters = {"group": group} if group is not None else {}
    runs = WandbAPI.get_handler().runs(
        config.get_wandb_project(), filters=filters, per_page=1000
    )

    summaries = []
    configs = []
    systems = []
    miscs = []

    for run in tqdm(runs):
        summaries.append(flatten_dict(run.summary._json_dict))
        configs.append(flatten_dict(run.config))
        systems.append(flatten_dict(run._attrs["systemMetrics"]))
        miscs.append(
            {
                "name": run.name,
                "id": run.id,
                "group": run.group,
                "state": run.state,
                "tags": run.tags,
                "histLineCount": run._attrs["historyLineCount"],
            }
        )

        if download_runs:
            if only_new:
                run_exists = os.path.isfile(get_run_path(run.id))
                if not run_exists:
                    download_run(run)
            else:
                download_run(run)

    misc_df = pd.DataFrame.from_records(miscs)
    summary_df = pd.DataFrame.from_records(summaries)
    config_df = pd.DataFrame.from_records(configs)
    system_df = pd.DataFrame.from_records(systems)
    all_df = pd.concat([misc_df, config_df, summary_df, system_df], axis=1)

    all_df.to_csv(SUMMARY_FILE)


##
# Data pre-processing


def concatenate_all_runs():
    """Concatenates all run .csv files into one file."""
    summary_df = get_summary()

    dfs = []
    ids = list(summary_df.index)
    for id in tqdm(ids):
        dfs.append(get_run(id))

    concat_df = pd.concat(dfs, keys=ids)
    concat_df.index = concat_df.index.set_names(names=["id", "step"])
    concat_df.to_csv(ALLRUNS_FILE)


def filter_out_tags(df, fromlist=None):
    """Removes lines from a dataframe if they are tagged.

    Filters out all tags by default. If fromlist is a list of strings,
    filters out only those tags.
    """
    new_df = df

    def filter_any(taglist):
        return len(taglist) > 0

    def filter_fromlist(taglist):
        return any([tag_to_filter in taglist for tag_to_filter in fromlist])

    filter = filter_any if fromlist is None else filter_fromlist

    def should_filter_out(row):
        s_ = str(row["tags"])
        s_ = s_.replace("'", '"')
        as_list = json.loads(s_)
        return filter(as_list)

    should_filter = new_df.apply(func=should_filter_out, axis=1)

    return df[~should_filter]


##
# Helper functions


def df_foreach(df, column, sortfunc=sorted):
    """Iterates through subsets of df for each unique value of column."""
    for unique_value in sortfunc(df[column].unique()):
        yield unique_value, df[df[column] == unique_value]


def df_select(df, **kwargs):
    """Select subsets of the dataframe by key/value pairs in kwargs."""

    if len(kwargs) == 0:
        return df

    def makemask(column, value):
        if value is None:
            return column.isna()
        elif isinstance(value, float):
            if 0 < value and value < 1e-5:
                return np.isclose(column, value, atol=0)
            else:
                return np.isclose(column, value)

        else:
            return column == value

    selection_masks = [makemask(df[k], v) for k, v in kwargs.items()]
    mask = selection_masks[0]
    for newmask in selection_masks[1:]:
        mask &= newmask
    return df[mask]


def df_unselect(df, **kwargs):
    """Select subsets of the dataframe by key/value pairs in kwargs."""

    if len(kwargs) == 0:
        return df

    def makemask(column, value):
        if value is None:
            return not column.isna()
        elif isinstance(value, float):
            return np.isclose(column, value) == False
        else:
            return column != value

    selection_masks = [makemask(df[k], v) for k, v in kwargs.items()]
    mask = selection_masks[0]
    for newmask in selection_masks[1:]:
        mask &= newmask
    return df[mask]


def median_min_max(df, key):
    return df[key].median(), df[key].min(), df[key].max()


def median_min_max_by(dataframe, key, metric_name):
    sub_df = dataframe[[key, metric_name]]
    groupby = sub_df.groupby(key)

    transforms_df = groupby.agg(["min", "max", "median"])

    medians = np.array(transforms_df[metric_name]["median"])
    mins = np.array(transforms_df[metric_name]["min"])
    maxs = np.array(transforms_df[metric_name]["max"])
    xs = np.array(transforms_df.index)

    return medians, mins, maxs, xs


def make_mask(df, selections: List[Dict[str, Any]]):
    def makemask(column, value):
        if value is None:
            return column.isna()
        elif isinstance(value, float):
            if 0 < value and value < 1e-5:
                return np.isclose(column, value, atol=0)
            else:
                return np.isclose(column, value)
        else:
            return column == value

    def make_mask_for_dict(selection_dict: Dict[str, Any]):
        selection_masks = [makemask(df[k], v) for k, v in selection_dict.items()]
        mask = selection_masks[0]
        for newmask in selection_masks[1:]:
            mask &= newmask
        return mask

    selection_mask = make_mask_for_dict(selections[0])
    for new_selection_mask in [make_mask_for_dict(_) for _ in selections[1:]]:
        selection_mask |= new_selection_mask

    return selection_mask


def new_select(df, selections: List[Dict[str, Any]]):
    """Select subsets of the dataframe by key/value pairs.

    selections is a list of dictionaries.
    The dictionaries are ORed while their elements are ANDed.

    Example:
        selection = [
            {"dataset": "mnist", "optim": "SGD"},
            {"dataset": "mnist", "optim": "Adam"}
        ]
        selects for
            (
                (dataset == mnist and optim == SGD)
                or (dataset == mnist and optim == Adam)
            )
    """
    if len(selections) == 0:
        return df

    return df[make_mask(df, selections)]


def new_filter_and_merge(summary, runs, summary_filter, runs_filter):
    summary_filtered = new_select(summary, summary_filter)
    runs_filtered = new_select(runs, runs_filter)
    runs_filtered = runs_filtered.loc[runs_filtered["id"].isin(summary_filtered.index)]

    ##
    # Drop duplicate columns

    run_columns_to_drop = [
        "grad_updates_per_epoch",
        "minibatches_per_epoch",
        "exp_runtime_s",
        "dataset",
    ]
    runs_filtered = runs_filtered.drop(labels=run_columns_to_drop, axis=1)

    merged = runs_filtered.join(
        other=summary_filtered, on="id", how="right", rsuffix="_end"
    )

    return merged


def flatten(t):
    return [item for sublist in t for item in sublist]


def grid_search(data, setting, opt, epoch, metric, key):
    """Grid search.

    for (settings AND opt AND epoch)
        find best value for key
            where "best" is minimum of (maximum across runs with that value)

    return dataframes:
        all runs (settings AND opt) at EPOCH
        best runs (settings AND opt AND key=best) for all epochs
    """
    setting_mask = make_mask(data, setting)
    opt_mask = make_mask(data, [opt])
    epoch_mask = make_mask(data, [{"epoch": epoch}])

    all_runs_at_epoch = data[setting_mask & opt_mask & epoch_mask]

    meds, mins, maxs, xs = median_min_max_by(
        all_runs_at_epoch, key=key, metric_name=metric
    )

    best_value = xs[np.nanargmin(maxs)]

    best_value_mask = make_mask(data, [{key: best_value}])

    best_runs_all_epochs = data[setting_mask & opt_mask & best_value_mask]

    return all_runs_at_epoch, best_runs_all_epochs


def gridsearch_for(data, dss, bss, opts, epoch_clip, experiments=expdef.EXPERIMENTS):
    runs_at_last_epoch_list = []
    best_runs_list = []

    for ds in dss:
        for bs in bss:
            for opt in opts:
                runs_at_last_epoch_, best_runs_ = grid_search(
                    data,
                    setting=experiments[ds][bs],
                    opt=expdef.OPTIMS[opt],
                    epoch=epoch_clip[ds][bs],
                    metric="training_loss",
                    key="opt.alpha",
                )
                runs_at_last_epoch_list.append(runs_at_last_epoch_)
                best_runs_list.append(best_runs_)

    runs_at_last_epoch = pd.concat(runs_at_last_epoch_list)
    best_runs = pd.concat(best_runs_list)
    return runs_at_last_epoch, best_runs


def add_stop_at_info(dataframe, stop_at):
    if any(["epoch_to_stop" == key for key in list(dataframe.keys())]):
        return dataframe

    dataframe["eff_bs"] = dataframe["batch_size"] * dataframe["accumulate_steps"]
    epoch_stop_table = pd.DataFrame(
        [
            {
                "dataset": ds,
                "eff_bs": expdef.EFF_BS[ds][bs],
                "epoch_to_stop": stop_at[ds][bs],
            }
            for ds in [
                expdef.MNIST,
                expdef.CIFAR10,
                expdef.PTB,
                expdef.WT2,
                expdef.SQUAD,
            ]
            for bs in [expdef.S, expdef.M, expdef.L, expdef.XL, expdef.FULL]
        ]
    )
    return pd.merge(left=dataframe, right=epoch_stop_table, on=["dataset", "eff_bs"])
