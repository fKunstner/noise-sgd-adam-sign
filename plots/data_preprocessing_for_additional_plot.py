import os
import pdb
import pickle

import explib.results.cleanup as cleanh
import explib.results.data as datah
import explib.results.plotting as plth
import numpy as np
from explib import config
from tqdm import tqdm


def standard_gridsearch():
    df, runs = datah.get_summary(), datah.get_all_runs()
    df, runs = cleanh.clean_data(df, runs)

    metric_types = ["training_loss", "training_perf", "validation_perf"]
    optimizers = plth.opt_names
    batch_sizes = {"small": 0, "medium": 1, "large": 2, "larger": 3, "full": None}

    def load_data(problem_slug):
        dataset = plth.problems[problem_slug]["dataset"]
        model = plth.problems[problem_slug]["model"]
        problem_filter = {"model": model, "dataset": dataset}

        summary = datah.df_select(df, **problem_filter)

        plot_data = {
            metric_type: {
                batch_size: {
                    opt: {
                        "value_at_init": None,
                        "ylims": None,
                        "alphas": None,
                        "best_alpha": None,
                        "best_alpha_idx": None,
                        "ys": None,
                        "ys+": None,
                        "ys-": None,
                        "batch_size": None,
                        "max_epoch": None,
                        "metric": None,
                    }
                    for opt in optimizers
                }
                for batch_size in batch_sizes.keys()
            }
            for metric_type in metric_types
        }

        for metric_type in tqdm(metric_types):
            metric = plth.metric_type_to_dset_to_metric[metric_type][dataset]
            metric_key = f"{metric}_runs"
            metric_value_at_init = plth.get_metric_at_start_for_dataset(
                runs, summary, dataset, metric
            )
            for batch_size_name, idx in tqdm(batch_sizes.items()):
                for opt in optimizers:
                    if batch_size_name == "full":
                        if problem_slug == "BRT":
                            experiment_filter = {"group": "full-batch-training-2"}
                        else:
                            experiment_filter = {"group": "full-batch-training"}
                        epoch_filter = {
                            "epoch": plth.experiment_settings["full_batch"][
                                "clip_epoch"
                            ][dataset]
                        }

                    else:
                        exp_settings = plth.experiment_settings["increasing_batch_size"]
                        experiment_filter = {
                            **exp_settings["problem_filters"][dataset][idx],
                            "group": "increasing-batch-size",
                        }
                        epoch_filter = {
                            "epoch": exp_settings["run_filters"][dataset][idx]["epoch"]
                        }

                    summary_filter = {
                        **problem_filter,
                        **experiment_filter,
                        **plth.opt_filters[opt],
                    }
                    runs_filter = {**epoch_filter}

                    runs_last_epoch = cleanh.filter_merge(
                        summary, runs, summary_filter, runs_filter
                    )

                    ylims = plth.get_metric_limits_for_dataset(
                        runs, summary, dataset, metric
                    )
                    ylims_trainingloss = plth.get_metric_limits_for_dataset(
                        runs, summary, dataset, "training_loss"
                    )

                    runs_last_epoch = plth.clip_metric_at(
                        runs_last_epoch, metric_key, ylims
                    )
                    runs_last_epoch = plth.clip_metric_at(
                        runs_last_epoch, "training_loss_runs", ylims_trainingloss
                    )
                    medians, mins, maxs, alphas = datah.median_min_max_by(
                        runs_last_epoch, key="opt.alpha", metric_name=metric_key
                    )
                    best_alpha = plth.find_best_stepsize(
                        runs_last_epoch, by_key="training_loss_runs"
                    )

                    best_alpha_idx = np.where(alphas == best_alpha)[0][0]

                    if batch_size_name == "full":
                        batch_size = "Full"
                    else:
                        batch_size = experiment_filter["batch_size"]
                        batch_size *= experiment_filter.get("accumulate_steps", 1)

                    plot_data[metric_type][batch_size_name][opt] = {
                        "metric": metric,
                        "value_at_init": metric_value_at_init,
                        "ylims": ylims,
                        "alphas": alphas,
                        "best_alpha": best_alpha,
                        "best_alpha_idx": best_alpha_idx,
                        "ys": medians,
                        "ys+": maxs,
                        "ys-": mins,
                        "batch_size": batch_size,
                        "max_epoch": epoch_filter.get("epoch"),
                    }
        return plot_data

    problem_slugs = ["LEN", "RES", "TEC", "TXL", "BRT"]
    for slug in problem_slugs:
        plth.save_preprocessed(load_data(slug), f"gridsearch_{slug}.pk")


def normalized_gridsearch():
    metric_types = ["training_loss", "training_perf", "validation_perf"]
    optimizers = [
        "NormalizedGD",
        "SignDescent",
        "RescaledSignDescent",
        "NormalizedGD+m",
        "SignDescent+m",
        "RescaledSignDescent+m",
    ]
    batch_sizes = ["medium", "large", "larger", "full"]

    df, runs = datah.get_summary(), datah.get_all_runs()
    df, runs = cleanh.clean_data(df, runs)

    def load_data(problem_slug):
        summary = df

        dataset = plth.problems[problem_slug]["dataset"]
        model = plth.problems[problem_slug]["model"]
        problem_filter = {"model": model, "dataset": dataset}

        summary = datah.df_select(summary, **problem_filter)

        plot_data = {
            metric_type: {
                batch_size: {
                    opt: {
                        "value_at_init": None,
                        "ylims": None,
                        "alphas": None,
                        "best_alpha": None,
                        "best_alpha_idx": None,
                        "ys": None,
                        "ys+": None,
                        "ys-": None,
                        "max_epoch": None,
                        "metric": None,
                    }
                    for opt in optimizers
                }
                for batch_size in batch_sizes
            }
            for metric_type in metric_types
        }

        for metric_type in tqdm(metric_types):
            metric = plth.metric_type_to_dset_to_metric[metric_type][dataset]
            metric_key = f"{metric}_runs"
            metric_value_at_init = plth.get_metric_at_start_for_dataset(
                runs, summary, dataset, metric
            )
            for batch_size_name in tqdm(batch_sizes):
                for opt in optimizers:

                    # if "+m" in opt and problem_slug == "BRT":
                    #     continue
                    # if batch_size_name == "full" and "Rescaled" in opt:
                    #     continue

                    if batch_size_name == "full":
                        exp_set = plth.experiment_settings["norm-ablation-full"]
                        group = "full-batch-training-normalized-optimizers"
                        max_epoch = exp_set["max_epoch"][dataset]
                        clip_epoch = exp_set["clip_epoch"][dataset]

                        exp_filter = {"group": group, "max_epoch": max_epoch}
                    else:
                        exp_set = plth.experiment_settings["norm-ablation"]
                        exp_set = exp_set[dataset][batch_size_name]
                        group = "normalization-ablation"
                        max_epoch = exp_set["max_epoch"]
                        clip_epoch = exp_set["clip_epoch"]
                        batch_size = exp_set["batch_size"]

                        exp_filter = {
                            "group": group,
                            "max_epoch": max_epoch,
                            "batch_size": batch_size,
                        }

                        if "accumulate_steps" in exp_set:
                            exp_filter["accumulate_steps"] = exp_set["accumulate_steps"]

                    epoch_filter = {"epoch": clip_epoch}

                    summary_filter = {
                        **problem_filter,
                        **exp_filter,
                        **plth.opt_filters[opt],
                    }
                    runs_filter = {**epoch_filter}

                    runs_last_epoch = cleanh.filter_merge(
                        summary, runs, summary_filter, runs_filter
                    )

                    # %%
                    # Find ylimits and truncate metric to worst-case if nan

                    ylims = plth.get_metric_limits_for_dataset(
                        runs, summary, dataset, metric
                    )
                    ylims_trainingloss = plth.get_metric_limits_for_dataset(
                        runs, summary, dataset, "training_loss"
                    )
                    runs_last_epoch = plth.clip_metric_at(
                        runs_last_epoch, metric_key, ylims
                    )
                    runs_last_epoch = plth.clip_metric_at(
                        runs_last_epoch, "training_loss_runs", ylims_trainingloss
                    )

                    # %%
                    # Find best step-size

                    (medians, mins, maxs, alphas,) = datah.median_min_max_by(
                        runs_last_epoch, key="opt.alpha", metric_name=metric_key
                    )
                    try:
                        best_alpha = plth.find_best_stepsize(
                            runs_last_epoch, by_key="training_loss_runs"
                        )
                    except:
                        import pdb

                        pdb.set_trace()

                    best_alpha_idx = np.where(alphas == best_alpha)[0][0]

                    plot_data[metric_type][batch_size_name][opt] = {
                        "metric": metric,
                        "value_at_init": metric_value_at_init,
                        "ylims": ylims,
                        "alphas": alphas,
                        "best_alpha": best_alpha,
                        "best_alpha_idx": best_alpha_idx,
                        "ys": medians,
                        "ys+": maxs,
                        "ys-": mins,
                        "max_epoch": epoch_filter.get("epoch"),
                    }
        return plot_data

    problem_slugs = ["LEN", "RES", "TEC", "TXL", "BRT"]
    for slug in problem_slugs:
        plth.save_preprocessed(load_data(slug), f"normalized_gridsearch_{slug}.pk")


def best_runs_for_each_batch_normalized_optimizers():
    metric_types = ["training_loss", "training_perf", "validation_perf"]
    optimizers = [
        "NormalizedGD",
        "SignDescent",
        "RescaledSignDescent",
        "NormalizedGD+m",
        "SignDescent+m",
        "RescaledSignDescent+m",
    ]
    batch_sizes = ["medium", "large", "larger", "full"]

    df, runs = datah.get_summary(), datah.get_all_runs()
    df, runs = cleanh.clean_data(df, runs)

    def load_data(problem_slug):
        dataset = plth.problems[problem_slug]["dataset"]
        model = plth.problems[problem_slug]["model"]
        problem_filter = {"model": model, "dataset": dataset}

        summary = datah.df_select(df, **problem_filter)

        plot_data = {
            metric_type: {
                batch_size: {
                    opt: {
                        "runs_wuuid": None,
                        "value_at_init": None,
                        "ylims": None,
                        "best_alpha": None,
                        "update_count": None,
                        "ys": None,
                        "ys+": None,
                        "ys-": None,
                        "max_epoch": None,
                        "metric": None,
                    }
                    for opt in optimizers
                }
                for batch_size in batch_sizes
            }
            for metric_type in metric_types
        }

        for metric_type in tqdm(metric_types):
            metric = plth.metric_type_to_dset_to_metric[metric_type][dataset]
            metric_key = f"{metric}_runs"
            metric_value_at_init = plth.get_metric_at_start_for_dataset(
                runs, summary, dataset, metric
            )
            for batch_size_name in tqdm(batch_sizes):
                for opt in optimizers:

                    # if "+m" in opt and problem_slug == "BRT":
                    #     continue
                    # if batch_size_name == "full" and "Rescaled" in opt:
                    #     continue

                    if batch_size_name == "full":
                        exp_set = plth.experiment_settings["norm-ablation-full"]
                        group = "full-batch-training-normalized-optimizers"
                        max_epoch = exp_set["max_epoch"][dataset]
                        clip_epoch = exp_set["clip_epoch"][dataset]

                        exp_filter = {"group": group, "max_epoch": max_epoch}
                    else:
                        exp_set = plth.experiment_settings["norm-ablation"]
                        exp_set = exp_set[dataset][batch_size_name]
                        group = "normalization-ablation"
                        max_epoch = exp_set["max_epoch"]
                        clip_epoch = exp_set["clip_epoch"]
                        batch_size = exp_set["batch_size"]

                        exp_filter = {
                            "group": group,
                            "max_epoch": max_epoch,
                            "batch_size": batch_size,
                        }

                        if "accumulate_steps" in exp_set:
                            exp_filter["accumulate_steps"] = exp_set["accumulate_steps"]

                    epoch_filter = {"epoch": clip_epoch}

                    summary_filter = {
                        **problem_filter,
                        **exp_filter,
                        **plth.opt_filters[opt],
                    }
                    runs_filter = {}

                    runs_ = cleanh.filter_merge(
                        summary, runs, summary_filter, runs_filter
                    )

                    # %%
                    # Truncate runs to specified number of epochs

                    runs_ = runs_[runs_["epoch_runs"] <= epoch_filter["epoch"]]

                    # %%
                    # Find ylimits and truncate metric to worst-case if nan

                    ylims = plth.get_metric_limits_for_dataset(
                        runs, summary, dataset, metric
                    )
                    ylims_trainingloss = plth.get_metric_limits_for_dataset(
                        runs, summary, dataset, "training_loss"
                    )
                    runs_ = plth.clip_metric_at(runs_, metric_key, ylims)
                    runs_ = plth.clip_metric_at(
                        runs_, "training_loss_runs", ylims_trainingloss
                    )

                    # %%
                    # Find best step-size and select runs

                    runs_at_lastepoch = runs_[
                        runs_["epoch_runs"] == epoch_filter["epoch"]
                    ]
                    best_alpha = plth.find_best_stepsize(
                        runs_at_lastepoch, by_key="training_loss_runs"
                    )
                    runs_ = datah.df_select(runs_, **{"opt.alpha": best_alpha})

                    medians, mins, maxs, update_count = datah.median_min_max_by(
                        runs_, key="update_count", metric_name=metric_key
                    )

                    (
                        medians_epoch,
                        mins_epoch,
                        maxs_epoch,
                        epoch,
                    ) = datah.median_min_max_by(
                        runs_, key="epoch_runs", metric_name=metric_key
                    )

                    detailed_info = {}
                    wuuids = list(runs_["wuuid"].unique())
                    for wuuid in wuuids:
                        runs_wuuid = datah.df_select(runs_, wuuid=wuuid)
                        runs_wuuid = runs_wuuid[
                            [
                                "step",
                                "norm_squared_gradients_runs",
                                "norm_squared_gradients_l1_runs",
                                "epoch_training_time_runs",
                            ]
                        ].sort_values("step")
                        detailed_info[wuuid] = {
                            "l2norms**2": list(
                                runs_wuuid["norm_squared_gradients_runs"]
                            ),
                            "l1norms**2": list(
                                runs_wuuid["norm_squared_gradients_l1_runs"]
                            ),
                            "runtimes": list(runs_wuuid["epoch_training_time_runs"]),
                        }

                    plot_data[metric_type][batch_size_name][opt] = {
                        "runs_wuuid": list(runs_["wuuid"].unique()),
                        "detailled_info": detailed_info,
                        "metric": metric,
                        "value_at_init": metric_value_at_init,
                        "ylims": ylims,
                        "best_alpha": best_alpha,
                        "update_count": update_count,
                        "ys": medians,
                        "ys+": maxs,
                        "ys-": mins,
                        "epochs": epoch,
                        "epochs_ys": medians_epoch,
                        "epochs_ys+": maxs_epoch,
                        "epochs_ys-": mins_epoch,
                        "max_epoch": epoch_filter.get("epoch"),
                    }
        return plot_data

    problem_slugs = ["LEN", "RES", "TEC", "TXL", "BRT"]
    # problem_slugs = ["TXL", "BRT"]
    for slug in problem_slugs:
        plth.save_preprocessed(load_data(slug), f"normalized_best_{slug}.pk")


def best_runs_for_each_batch_normal_optimizers():
    problem_slugs = ["LEN", "RES", "TEC", "TXL", "BRT"]
    metric_types = ["training_loss", "training_perf", "validation_perf"]
    optimizers = plth.opt_names
    batch_sizes = {"small": 0, "medium": 1, "large": 2, "larger": 3, "full": None}

    df, runs = datah.get_summary(), datah.get_all_runs()
    df, runs = cleanh.clean_data(df, runs)
    summary = df

    plot_data = {
        problem_slug: {
            metric_type: {
                batch_size: {
                    opt: {
                        "runs_wuuid": None,
                        "value_at_init": None,
                        "ylims": None,
                        "best_alpha": None,
                        "iter": None,
                        "ys": None,
                        "ys+": None,
                        "ys-": None,
                        "epochs": None,
                        "epochs_ys": None,
                        "epochs_ys+": None,
                        "epochs_ys-": None,
                        "batch_size": None,
                        "max_epoch": None,
                        "metric": None,
                    }
                    for opt in optimizers
                }
                for batch_size in batch_sizes.keys()
            }
            for metric_type in metric_types
        }
        for problem_slug in problem_slugs
    }

    datasets = [plth.problems[slug]["dataset"] for slug in problem_slugs]
    metrics_per_datasets = {
        dataset: [
            plth.metric_type_to_dset_to_metric[metric_type][dataset]
            for metric_type in metric_types
        ]
        for dataset in datasets
    }
    ylims_by_dataset_and_metric = {
        dataset: {
            metric: plth.get_metric_limits_for_dataset(runs, summary, dataset, metric)
            for metric in metrics_per_datasets[dataset]
        }
        for dataset in datasets
    }

    for problem_slug in tqdm(problem_slugs):
        dataset = plth.problems[problem_slug]["dataset"]
        model = plth.problems[problem_slug]["model"]
        problem_filter = {"model": model, "dataset": dataset}

        summary_problem = datah.df_select(summary, **problem_filter)

        for metric_type in metric_types:

            metric = plth.metric_type_to_dset_to_metric[metric_type][dataset]
            metric_key = f"{metric}_runs"
            metric_value_at_init = plth.get_metric_at_start_for_dataset(
                runs, summary, dataset, metric
            )

            ylims = ylims_by_dataset_and_metric[dataset][metric]
            ylims_trainingloss = ylims_by_dataset_and_metric[dataset]["training_loss"]

            for batch_size_name, idx in batch_sizes.items():
                if batch_size_name == "full":
                    if problem_slug == "BRT":
                        experiment_filter = {"group": "full-batch-training-2"}
                    else:
                        experiment_filter = {"group": "full-batch-training"}
                    epoch_filter = {
                        "epoch": plth.experiment_settings["full_batch"]["clip_epoch"][
                            dataset
                        ]
                    }

                else:
                    exp_settings = plth.experiment_settings["increasing_batch_size"]
                    experiment_filter = {
                        **exp_settings["problem_filters"][dataset][idx],
                        "group": "increasing-batch-size",
                    }
                    epoch_filter = {
                        "epoch": exp_settings["run_filters"][dataset][idx]["epoch"]
                    }

                summary_problem_bs = datah.df_select(
                    summary_problem, **experiment_filter
                )

                for opt in optimizers:

                    runs_ = cleanh.filter_merge(
                        summary_problem_bs, runs, plth.opt_filters[opt], {}
                    )

                    # %%
                    # Truncate runs to specified number of epochs

                    runs_ = runs_[runs_["epoch_runs"] <= epoch_filter["epoch"]]

                    # %%
                    # Find ylimits and truncate metric to worst-case if nan

                    runs_ = plth.clip_metric_at(runs_, metric_key, ylims)
                    runs_ = plth.clip_metric_at(
                        runs_, "training_loss_runs", ylims_trainingloss
                    )

                    # %%
                    # Find best step-size and select runs

                    runs_at_lastepoch = runs_[
                        runs_["epoch_runs"] == epoch_filter["epoch"]
                    ]
                    best_alpha = plth.find_best_stepsize(
                        runs_at_lastepoch, by_key="training_loss_runs"
                    )
                    runs_ = datah.df_select(runs_, **{"opt.alpha": best_alpha})

                    medians, mins, maxs, update_count = datah.median_min_max_by(
                        runs_, key="update_count", metric_name=metric_key
                    )

                    (
                        medians_epoch,
                        mins_epoch,
                        maxs_epoch,
                        epoch,
                    ) = datah.median_min_max_by(
                        runs_, key="epoch_runs", metric_name=metric_key
                    )

                    if batch_size_name == "full":
                        batch_size = "Full"
                    else:
                        batch_size = experiment_filter["batch_size"]
                        batch_size *= experiment_filter.get("accumulate_steps", 1)

                    try:
                        detailed_info = {}
                        wuuids = list(runs_["wuuid"].unique())
                        for wuuid in wuuids:
                            runs_wuuid = datah.df_select(runs_, wuuid=wuuid)
                            subset = runs_wuuid[
                                [
                                    "step",
                                    "norm_squared_gradients_runs",
                                    "norm_squared_gradients_l1_runs",
                                    "epoch_training_time_runs",
                                ]
                            ].sort_values("step")
                            detailed_info[wuuid] = {
                                "l2norms**2": list(
                                    subset["norm_squared_gradients_runs"]
                                ),
                                "l1norms**2": list(
                                    subset["norm_squared_gradients_l1_runs"]
                                ),
                                "runtimes": list(subset["epoch_training_time_runs"]),
                            }

                    except:
                        import pdb

                        pdb.set_trace()

                    plot_data[problem_slug][metric_type][batch_size_name][opt] = {
                        "runs_wuuid": list(runs_["wuuid"].unique()),
                        "detailled_info": detailed_info,
                        "metric": metric,
                        "value_at_init": metric_value_at_init,
                        "ylims": ylims,
                        "best_alpha": best_alpha,
                        "iter": update_count,
                        "ys": medians,
                        "ys+": maxs,
                        "ys-": mins,
                        "epochs": epoch,
                        "epochs_ys": medians_epoch,
                        "epochs_ys+": maxs_epoch,
                        "epochs_ys-": mins_epoch,
                        "batch_size": batch_size,
                        "max_epoch": epoch_filter.get("epoch"),
                    }

    plth.save_preprocessed(plot_data, "best_runs_for_each_batch_normal_optimizers.pk")


def gs_nodroupout():
    metric_types = ["training_loss", "training_perf", "validation_perf"]
    optimizers = plth.opt_names + [
        "NormalizedGD",
        "SignDescent",
        "RescaledSignDescent",
        "NormalizedGD+m",
        "SignDescent+m",
        "RescaledSignDescent+m",
    ]
    batch_sizes = ["full"]

    df, runs = datah.get_summary(), datah.get_all_runs()
    df, runs = cleanh.clean_data(df, runs)

    def load_data(problem_slug):
        summary = df

        dataset = plth.problems[problem_slug]["dataset"]
        problem_filter = {"dataset": dataset}

        summary = datah.df_select(summary, **problem_filter)

        plot_data = {
            metric_type: {
                batch_size: {
                    opt: {
                        "value_at_init": None,
                        "ylims": None,
                        "alphas": None,
                        "best_alpha": None,
                        "best_alpha_idx": None,
                        "ys": None,
                        "ys+": None,
                        "ys-": None,
                        "max_epoch": None,
                        "metric": None,
                    }
                    for opt in optimizers
                }
                for batch_size in batch_sizes
            }
            for metric_type in metric_types
        }

        for metric_type in tqdm(metric_types):
            metric = plth.metric_type_to_dset_to_metric[metric_type][dataset]
            metric_key = f"{metric}_runs"
            metric_value_at_init = plth.get_metric_at_start_for_dataset(
                runs, summary, dataset, metric
            )
            for batch_size_name in tqdm(batch_sizes):
                for opt in optimizers:

                    exp_set = plth.experiment_settings["no-dropout"]
                    exp_set = exp_set[dataset][batch_size_name]
                    group = "no-dropout"
                    max_epoch = exp_set["max_epoch"]
                    clip_epoch = exp_set["clip_epoch"]
                    batch_size = exp_set["batch_size"]

                    exp_filter = {
                        "group": group,
                        "max_epoch": max_epoch,
                        "batch_size": batch_size,
                    }

                    if "accumulate_steps" in exp_set:
                        exp_filter["accumulate_steps"] = exp_set["accumulate_steps"]

                    epoch_filter = {"epoch": clip_epoch}

                    summary_filter = {
                        **problem_filter,
                        **exp_filter,
                        **plth.opt_filters[opt],
                    }
                    runs_filter = {**epoch_filter}

                    runs_last_epoch = cleanh.filter_merge(
                        summary, runs, summary_filter, runs_filter
                    )

                    # %%
                    # Find ylimits and truncate metric to worst-case if nan

                    ylims = plth.get_metric_limits_for_dataset(
                        runs, summary, dataset, metric
                    )
                    ylims_trainingloss = plth.get_metric_limits_for_dataset(
                        runs, summary, dataset, "training_loss"
                    )
                    runs_last_epoch = plth.clip_metric_at(
                        runs_last_epoch, metric_key, ylims
                    )
                    runs_last_epoch = plth.clip_metric_at(
                        runs_last_epoch, "training_loss_runs", ylims_trainingloss
                    )

                    # %%
                    # Find best step-size

                    (medians, mins, maxs, alphas,) = datah.median_min_max_by(
                        runs_last_epoch, key="opt.alpha", metric_name=metric_key
                    )
                    try:
                        best_alpha = plth.find_best_stepsize(
                            runs_last_epoch, by_key="training_loss_runs"
                        )
                    except:
                        import pdb

                        pdb.set_trace()

                    best_alpha_idx = np.where(alphas == best_alpha)[0][0]

                    plot_data[metric_type][batch_size_name][opt] = {
                        "metric": metric,
                        "value_at_init": metric_value_at_init,
                        "ylims": ylims,
                        "alphas": alphas,
                        "best_alpha": best_alpha,
                        "best_alpha_idx": best_alpha_idx,
                        "ys": medians,
                        "ys+": maxs,
                        "ys-": mins,
                        "max_epoch": epoch_filter.get("epoch"),
                    }
        return plot_data

    problem_slugs = ["TEC", "TXL"]
    for slug in problem_slugs:
        plth.save_preprocessed(load_data(slug), f"gs_nodropout_{slug}.pk")


def best_runs_nodropout():
    metric_types = ["training_loss", "training_perf", "validation_perf"]
    optimizers = plth.opt_names + [
        "NormalizedGD",
        "SignDescent",
        "RescaledSignDescent",
        "NormalizedGD+m",
        "SignDescent+m",
        "RescaledSignDescent+m",
    ]
    batch_sizes = ["full"]

    df, runs = datah.get_summary(), datah.get_all_runs()
    df, runs = cleanh.clean_data(df, runs)

    def load_data(problem_slug):
        dataset = plth.problems[problem_slug]["dataset"]
        model = plth.problems[problem_slug]["model"]
        problem_filter = {"dataset": dataset}

        summary = datah.df_select(df, **problem_filter)

        plot_data = {
            metric_type: {
                batch_size: {
                    opt: {
                        "runs_wuuid": None,
                        "value_at_init": None,
                        "ylims": None,
                        "best_alpha": None,
                        "update_count": None,
                        "ys": None,
                        "ys+": None,
                        "ys-": None,
                        "max_epoch": None,
                        "metric": None,
                    }
                    for opt in optimizers
                }
                for batch_size in batch_sizes
            }
            for metric_type in metric_types
        }

        for metric_type in tqdm(metric_types):
            metric = plth.metric_type_to_dset_to_metric[metric_type][dataset]
            metric_key = f"{metric}_runs"
            metric_value_at_init = plth.get_metric_at_start_for_dataset(
                runs, summary, dataset, metric
            )
            for batch_size_name in tqdm(batch_sizes):
                for opt in optimizers:

                    exp_set = plth.experiment_settings["no-dropout"]
                    exp_set = exp_set[dataset][batch_size_name]
                    group = "no-dropout"
                    max_epoch = exp_set["max_epoch"]
                    clip_epoch = exp_set["clip_epoch"]
                    batch_size = exp_set["batch_size"]

                    exp_filter = {
                        "group": group,
                        "max_epoch": max_epoch,
                        "batch_size": batch_size,
                    }

                    if "accumulate_steps" in exp_set:
                        exp_filter["accumulate_steps"] = exp_set["accumulate_steps"]

                    epoch_filter = {"epoch": clip_epoch}

                    summary_filter = {
                        **problem_filter,
                        **exp_filter,
                        **plth.opt_filters[opt],
                    }
                    runs_filter = {}

                    runs_ = cleanh.filter_merge(
                        summary, runs, summary_filter, runs_filter
                    )

                    # %%
                    # Truncate runs to specified number of epochs

                    runs_ = runs_[runs_["epoch_runs"] <= epoch_filter["epoch"]]

                    # %%
                    # Find ylimits and truncate metric to worst-case if nan

                    ylims = plth.get_metric_limits_for_dataset(
                        runs, summary, dataset, metric
                    )
                    ylims_trainingloss = plth.get_metric_limits_for_dataset(
                        runs, summary, dataset, "training_loss"
                    )
                    runs_ = plth.clip_metric_at(runs_, metric_key, ylims)
                    runs_ = plth.clip_metric_at(
                        runs_, "training_loss_runs", ylims_trainingloss
                    )

                    # %%
                    # Find best step-size and select runs

                    runs_at_lastepoch = runs_[
                        runs_["epoch_runs"] == epoch_filter["epoch"]
                    ]
                    best_alpha = plth.find_best_stepsize(
                        runs_at_lastepoch, by_key="training_loss_runs"
                    )
                    runs_ = datah.df_select(runs_, **{"opt.alpha": best_alpha})

                    medians, mins, maxs, update_count = datah.median_min_max_by(
                        runs_, key="update_count", metric_name=metric_key
                    )

                    (
                        medians_epoch,
                        mins_epoch,
                        maxs_epoch,
                        epoch,
                    ) = datah.median_min_max_by(
                        runs_, key="epoch_runs", metric_name=metric_key
                    )

                    detailed_info = {}
                    wuuids = list(runs_["wuuid"].unique())
                    for wuuid in wuuids:
                        runs_wuuid = datah.df_select(runs_, wuuid=wuuid)
                        runs_wuuid = runs_wuuid[
                            [
                                "step",
                                "norm_squared_gradients_runs",
                                "norm_squared_gradients_l1_runs",
                                "epoch_training_time_runs",
                            ]
                        ].sort_values("step")
                        detailed_info[wuuid] = {
                            "l2norms**2": list(
                                runs_wuuid["norm_squared_gradients_runs"]
                            ),
                            "l1norms**2": list(
                                runs_wuuid["norm_squared_gradients_l1_runs"]
                            ),
                            "runtimes": list(runs_wuuid["epoch_training_time_runs"]),
                        }

                    plot_data[metric_type][batch_size_name][opt] = {
                        "runs_wuuid": list(runs_["wuuid"].unique()),
                        "detailled_info": detailed_info,
                        "metric": metric,
                        "value_at_init": metric_value_at_init,
                        "ylims": ylims,
                        "best_alpha": best_alpha,
                        "update_count": update_count,
                        "ys": medians,
                        "ys+": maxs,
                        "ys-": mins,
                        "epochs": epoch,
                        "epochs_ys": medians_epoch,
                        "epochs_ys+": maxs_epoch,
                        "epochs_ys-": mins_epoch,
                        "max_epoch": epoch_filter.get("epoch"),
                    }
        return plot_data

    problem_slugs = ["TEC", "TXL"]
    for slug in problem_slugs:
        plth.save_preprocessed(load_data(slug), f"nodropout_best_{slug}.pk")


def gs_fullbatch_squad_fix():
    metric_types = ["training_loss", "training_perf", "validation_perf"]
    optimizers = plth.opt_names + [
        "NormalizedGD",
        "SignDescent",
        #        "RescaledSignDescent",
        "NormalizedGD+m",
        "SignDescent+m",
        #        "RescaledSignDescent+m",
    ]
    batch_sizes = ["full"]

    df, runs = datah.get_summary(), datah.get_all_runs()
    df, runs = cleanh.clean_data(df, runs)

    def load_data(problem_slug):
        summary = df

        dataset = plth.problems[problem_slug]["dataset"]
        problem_filter = {"dataset": dataset}

        summary = datah.df_select(summary, **problem_filter)

        plot_data = {
            metric_type: {
                batch_size: {
                    opt: {
                        "value_at_init": None,
                        "ylims": None,
                        "alphas": None,
                        "best_alpha": None,
                        "best_alpha_idx": None,
                        "ys": None,
                        "ys+": None,
                        "ys-": None,
                        "max_epoch": None,
                        "metric": None,
                    }
                    for opt in optimizers
                }
                for batch_size in batch_sizes
            }
            for metric_type in metric_types
        }

        for metric_type in tqdm(metric_types):
            metric = plth.metric_type_to_dset_to_metric[metric_type][dataset]
            metric_key = f"{metric}_runs"
            metric_value_at_init = plth.get_metric_at_start_for_dataset(
                runs, summary, dataset, metric
            )
            for batch_size_name in tqdm(batch_sizes):
                for opt in optimizers:

                    exp_set = plth.experiment_settings["fix-full-batch-training-squad"]
                    exp_set = exp_set[dataset][batch_size_name]
                    group = "fix-full-batch-training-squad"
                    max_epoch = exp_set["max_epoch"]
                    clip_epoch = exp_set["clip_epoch"]
                    batch_size = exp_set["batch_size"]

                    exp_filter = {
                        "group": group,
                        "max_epoch": max_epoch,
                        "batch_size": batch_size,
                    }

                    if "accumulate_steps" in exp_set:
                        exp_filter["accumulate_steps"] = exp_set["accumulate_steps"]

                    epoch_filter = {"epoch": clip_epoch}

                    summary_filter = {
                        **problem_filter,
                        **exp_filter,
                        **plth.opt_filters[opt],
                    }
                    runs_filter = {**epoch_filter}

                    runs_last_epoch = cleanh.filter_merge(
                        summary, runs, summary_filter, runs_filter
                    )

                    # %%
                    # Find ylimits and truncate metric to worst-case if nan

                    ylims = plth.get_metric_limits_for_dataset(
                        runs, summary, dataset, metric
                    )
                    ylims_trainingloss = plth.get_metric_limits_for_dataset(
                        runs, summary, dataset, "training_loss"
                    )
                    runs_last_epoch = plth.clip_metric_at(
                        runs_last_epoch, metric_key, ylims
                    )
                    runs_last_epoch = plth.clip_metric_at(
                        runs_last_epoch, "training_loss_runs", ylims_trainingloss
                    )

                    # %%
                    # Find best step-size

                    (medians, mins, maxs, alphas,) = datah.median_min_max_by(
                        runs_last_epoch, key="opt.alpha", metric_name=metric_key
                    )
                    try:
                        best_alpha = plth.find_best_stepsize(
                            runs_last_epoch, by_key="training_loss_runs"
                        )
                    except:
                        import pdb

                        pdb.set_trace()

                    best_alpha_idx = np.where(alphas == best_alpha)[0][0]

                    plot_data[metric_type][batch_size_name][opt] = {
                        "metric": metric,
                        "value_at_init": metric_value_at_init,
                        "ylims": ylims,
                        "alphas": alphas,
                        "best_alpha": best_alpha,
                        "best_alpha_idx": best_alpha_idx,
                        "ys": medians,
                        "ys+": maxs,
                        "ys-": mins,
                        "max_epoch": epoch_filter.get("epoch"),
                    }
        return plot_data

    problem_slugs = ["BRT"]
    for slug in problem_slugs:
        plth.save_preprocessed(load_data(slug), f"gs_squadfix_{slug}.pk")


def best_runs_fullbatch_squad_fix():
    metric_types = ["training_loss", "training_perf", "validation_perf"]
    optimizers = plth.opt_names + [
        "NormalizedGD",
        "SignDescent",
        #        "RescaledSignDescent",
        "NormalizedGD+m",
        "SignDescent+m",
        #        "RescaledSignDescent+m",
    ]
    batch_sizes = ["full"]

    df, runs = datah.get_summary(), datah.get_all_runs()
    df, runs = cleanh.clean_data(df, runs)

    def load_data(problem_slug):
        dataset = plth.problems[problem_slug]["dataset"]
        model = plth.problems[problem_slug]["model"]
        problem_filter = {"dataset": dataset}

        summary = datah.df_select(df, **problem_filter)

        plot_data = {
            metric_type: {
                batch_size: {
                    opt: {
                        "runs_wuuid": None,
                        "value_at_init": None,
                        "ylims": None,
                        "best_alpha": None,
                        "update_count": None,
                        "ys": None,
                        "ys+": None,
                        "ys-": None,
                        "max_epoch": None,
                        "metric": None,
                    }
                    for opt in optimizers
                }
                for batch_size in batch_sizes
            }
            for metric_type in metric_types
        }

        for metric_type in tqdm(metric_types):
            metric = plth.metric_type_to_dset_to_metric[metric_type][dataset]
            metric_key = f"{metric}_runs"
            metric_value_at_init = plth.get_metric_at_start_for_dataset(
                runs, summary, dataset, metric
            )
            for batch_size_name in tqdm(batch_sizes):
                for opt in optimizers:

                    exp_set = plth.experiment_settings["fix-full-batch-training-squad"]
                    exp_set = exp_set[dataset][batch_size_name]
                    group = "fix-full-batch-training-squad"
                    max_epoch = exp_set["max_epoch"]
                    clip_epoch = exp_set["clip_epoch"]
                    batch_size = exp_set["batch_size"]

                    exp_filter = {
                        "group": group,
                        "max_epoch": max_epoch,
                        "batch_size": batch_size,
                    }

                    if "accumulate_steps" in exp_set:
                        exp_filter["accumulate_steps"] = exp_set["accumulate_steps"]

                    epoch_filter = {"epoch": clip_epoch}

                    summary_filter = {
                        **problem_filter,
                        **exp_filter,
                        **plth.opt_filters[opt],
                    }
                    runs_filter = {}

                    runs_ = cleanh.filter_merge(
                        summary, runs, summary_filter, runs_filter
                    )

                    # %%
                    # Truncate runs to specified number of epochs

                    runs_ = runs_[runs_["epoch_runs"] <= epoch_filter["epoch"]]

                    # %%
                    # Find ylimits and truncate metric to worst-case if nan

                    ylims = plth.get_metric_limits_for_dataset(
                        runs, summary, dataset, metric
                    )
                    ylims_trainingloss = plth.get_metric_limits_for_dataset(
                        runs, summary, dataset, "training_loss"
                    )
                    runs_ = plth.clip_metric_at(runs_, metric_key, ylims)
                    runs_ = plth.clip_metric_at(
                        runs_, "training_loss_runs", ylims_trainingloss
                    )

                    # %%
                    # Find best step-size and select runs

                    runs_at_lastepoch = runs_[
                        runs_["epoch_runs"] == epoch_filter["epoch"]
                    ]
                    try:
                        best_alpha = plth.find_best_stepsize(
                            runs_at_lastepoch, by_key="training_loss_runs"
                        )
                        runs_ = datah.df_select(runs_, **{"opt.alpha": best_alpha})
                    except:
                        import pdb

                        pdb.set_trace()

                    medians, mins, maxs, update_count = datah.median_min_max_by(
                        runs_, key="update_count", metric_name=metric_key
                    )

                    (
                        medians_epoch,
                        mins_epoch,
                        maxs_epoch,
                        epoch,
                    ) = datah.median_min_max_by(
                        runs_, key="epoch_runs", metric_name=metric_key
                    )

                    detailed_info = {}
                    wuuids = list(runs_["wuuid"].unique())
                    for wuuid in wuuids:
                        runs_wuuid = datah.df_select(runs_, wuuid=wuuid)
                        runs_wuuid = runs_wuuid[
                            [
                                "step",
                                "norm_squared_gradients_runs",
                                "norm_squared_gradients_l1_runs",
                                "epoch_training_time_runs",
                            ]
                        ].sort_values("step")
                        detailed_info[wuuid] = {
                            "l2norms**2": list(
                                runs_wuuid["norm_squared_gradients_runs"]
                            ),
                            "l1norms**2": list(
                                runs_wuuid["norm_squared_gradients_l1_runs"]
                            ),
                            "runtimes": list(runs_wuuid["epoch_training_time_runs"]),
                        }

                    plot_data[metric_type][batch_size_name][opt] = {
                        "runs_wuuid": list(runs_["wuuid"].unique()),
                        "detailled_info": detailed_info,
                        "metric": metric,
                        "value_at_init": metric_value_at_init,
                        "ylims": ylims,
                        "best_alpha": best_alpha,
                        "update_count": update_count,
                        "ys": medians,
                        "ys+": maxs,
                        "ys-": mins,
                        "epochs": epoch,
                        "epochs_ys": medians_epoch,
                        "epochs_ys+": maxs_epoch,
                        "epochs_ys-": mins_epoch,
                        "max_epoch": epoch_filter.get("epoch"),
                    }
        return plot_data

    problem_slugs = ["BRT"]
    for slug in problem_slugs:
        plth.save_preprocessed(load_data(slug), f"squadfix_best_{slug}.pk")


if __name__ == "__main__":
    best_runs_fullbatch_squad_fix()
