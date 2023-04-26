from explib.expmaker.experiment_defs import load_summary
from explib.expmaker.experiment_defs import make_wuuid
from explib.results.cleanup import process_tags
import explib.results.data as data_h
import pandas as pd
from functools import lru_cache
import json


def wuuid_to_successful_run(exp_name):
    """Returns a dictionary of uuid: boolean indicating whether that experiment
    UUID has one succesful run on Wandb.
    """
    df = load_summary_and_wandb_success(exp_name)
    newdict = {}
    for row in df.to_dict(orient="records"):
        newdict[row["wuuid"]] = row["has_a_finished_run"]
    return newdict


@lru_cache(maxsize=None, typed=False)
def load_summary_and_wandb_success(exp_name, download=True):
    """Returns the load_summary for exp_name augmented with a
    "has_a_finished_run" column that checks for Wandb completion."""

    local_summary = load_summary(exp_name)
    local_df = pd.DataFrame(
        [
            {"uuid": k, "wuuid": make_wuuid(exp_dict), **data_h.flatten_dict(exp_dict)}
            for k, exp_dict in local_summary.items()
        ]
    )

    if download:
        data_h.download_summary()
    wandb_df = data_h.get_summary(ignore_per_iter_data=True)
    wandb_df["success"] = wandb_df["status"] == "Success"
    wandb_df["finished"] = wandb_df["status"] == "Finished with no summary"
    wandb_df["diverged"] = wandb_df["status"] == "Diverged"

    wandb_df, _ = process_tags(wandb_df, None)

    wandb_df["finished"] = (
        wandb_df["success"] | wandb_df["diverged"] | wandb_df["finished"]
    )
    wandb_df["success"] = wandb_df["finished"] & ~wandb_df["bad_tag"]

    wandb_success_df = (
        wandb_df.groupby("wuuid")["success"]
        .any()
        .rename("has_a_finished_run")
        .to_frame()
    )

    merged_df = pd.merge(
        left=local_df,
        left_on="wuuid",
        right=wandb_success_df,
        right_on="wuuid",
        how="left",
    )

    merged_df["has_a_finished_run"] = merged_df["has_a_finished_run"].fillna(False)
    return merged_df


def check_status(exp_name, hyperparam_names=None, download=False):
    """
    Prints the % of runs that have finished on Wandb.
    If hyperparam_names is a list of hyperparameter names in wandb format
    (eg "opt.b1"), breaks it down per hyperparam.
    """
    merged_df = load_summary_and_wandb_success(exp_name, download=download)
    print(f"Total completion; {merged_df['has_a_finished_run'].mean()*100:.2f}%")
    if hyperparam_names is not None:
        for colname in hyperparam_names:
            print(f"  Completion for [{colname}]")
            count = merged_df.groupby(colname)["has_a_finished_run"].count()
            avg_finished = merged_df.groupby(colname)["has_a_finished_run"].mean()
            for key in sorted(list(avg_finished.keys())):
                print(
                    f"    {key:>24} : {avg_finished[key]*100: >#04.2f}% (out of {count[key]})"
                )
