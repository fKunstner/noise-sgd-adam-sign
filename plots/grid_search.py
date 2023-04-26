import importlib
import os
from pathlib import Path

import explib.results.cleanup as cleanh
import explib.results.data as data_h
import explib.results.data as datah
import explib.results.data_caching as data_cache
import explib.results.experiment_groups as expdef
import explib.results.plotting as plth
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker


def load_data():
    importlib.reload(datah)
    importlib.reload(cleanh)
    importlib.reload(plth)

    runs_at_last_epoch, best_runs = data_cache.gridsearch_all_end()
    data = data_cache.load_filtered_data()

    return {"data": data, "best_runs": best_runs, "last_epoch": runs_at_last_epoch}


def settings(plt):
    importlib.reload(datah)
    importlib.reload(cleanh)
    importlib.reload(plth)

    plt.rcParams.update(plth.iclr_config_2(nrows=3, ncols=5, height_to_width_ratio=1))


def make_figure(fig, data, dataset=expdef.SQUAD, opts_to_plot="normalized"):
    importlib.reload(datah)
    importlib.reload(cleanh)
    importlib.reload(plth)
    importlib.reload(expdef)

    last_epoch = data["last_epoch"]
    best_runs = data["best_runs"]
    all_data = data["data"]
    epoch_filter = [{"epoch": 0}]
    all_data_at_start = data_h.new_select(all_data, epoch_filter)

    if opts_to_plot == "standard":
        opts = [expdef.SGD_NM, expdef.ADAM_NM, expdef.SGD_M, expdef.ADAM_M]
    elif opts_to_plot == "normalized":
        opts = [expdef.SIGN_NM, expdef.NORM_NM, expdef.SIGN_M, expdef.NORM_M]
    else:
        raise ValueError(f"Unknown opts {opts_to_plot}")

    bss = expdef.ALL_BS
    grid_type = "3x5"
    metrics = ["training_loss", "training_perf", "validation_perf"]
    metrics_for_ds = [
        plth.metric_type_to_dset_to_metric[metric][dataset] for metric in metrics
    ]
    axes = plth.make_grid_iclr(fig, grid_type=grid_type, tight=True)

    def get_data(ds, bs, opt):
        bs_filter = expdef.EXPERIMENTS[dataset][bs]
        last_epoch_for_ds = data_h.new_select(last_epoch, bs_filter)
        all_data_at_start_for_ds = data_h.new_select(all_data_at_start, bs_filter)
        best_runs_for_ds = data_h.new_select(best_runs, bs_filter)

        opt_filter = [expdef.OPTIMS[opt]]
        last_epoch_ = data_h.new_select(last_epoch_for_ds, opt_filter)
        best_runs_ = data_h.new_select(best_runs_for_ds, opt_filter)
        all_data_at_start_ = data_h.new_select(all_data_at_start_for_ds, opt_filter)

        step_size_perf = last_epoch_.groupby("opt.alpha")[metrics_for_ds].agg(
            [min, max, "median"]
        )
        best_ss = best_runs_["opt.alpha"].unique()[0]

        all_data_at_start_ = all_data_at_start_.copy()
        all_data_at_start_["has_diverged"] = all_data_at_start_["status"] == "Diverged"
        ss_diverging_status = (
            all_data_at_start_[["opt.alpha", "has_diverged"]]
            .groupby("opt.alpha")
            .agg("any")
        )

        step_size_perf.columns = step_size_perf.columns.to_flat_index()

        ss_perf = pd.merge(
            step_size_perf,
            ss_diverging_status,
            left_index=True,
            right_index=True,
            how="outer",
        )

        for col in ss_perf.keys():
            if "accuracy" in col[0] or "f1" in col[0]:
                ss_perf[col] = ss_perf[col].fillna(value=0)
            else:
                ss_perf[col] = ss_perf[col].fillna(
                    value=(10**4) * ss_perf[col].median()
                )
        ss_perf = ss_perf.drop("has_diverged", axis=1)

        return ss_perf, best_ss, all_data_at_start_

    for i, bs in enumerate(bss):
        for opt in opts:
            step_size_perf, best_ss, at_start = get_data(ds=dataset, bs=bs, opt=opt)
            for j, metric in enumerate(metrics_for_ds):
                ax = axes[j][i]

                linestyle = plth.linestyles[opt].copy()
                ax.plot(
                    step_size_perf.index,
                    step_size_perf[(metric, "median")],
                    **linestyle,
                )
                ax.fill_between(
                    step_size_perf.index,
                    step_size_perf[(metric, "min")],
                    step_size_perf[(metric, "max")],
                    **plth.fillstyles[opt],
                )
                linestyle["marker"] = "*"
                linestyle["markersize"] = "4"
                linestyle["color"] = "k"

                ax.plot(
                    best_ss,
                    step_size_perf[(metric, "median")][best_ss],
                    **linestyle,
                    zorder=10,
                )
                ax.axhline(
                    at_start[metric].median(),
                    linewidth=plth.linewidth_small,
                    color=plth.BASE_COLORS["gray"],
                    label="Init.",
                    zorder=-10,
                )

    ylims_trainingloss = {
        expdef.MNIST: {
            "lims": [10**-6, 10**2],
            "ticks": [-5, -3, -1],
        },
        expdef.CIFAR10: {
            "lims": [10**-7, 10**2],
            "ticks": [-6, -3, 0],
        },
        expdef.PTB: {"lims": [1.0, 12], "ticks": [2, 4, 6]},
        expdef.WT2: {"lims": [10**-1, 10**1.5], "ticks": [-1, 0, 1.0]},
        expdef.SQUAD: {"lims": [10**-1.5, 10**1.0], "ticks": [-1, 0]},
    }
    ylims_PPL = {
        expdef.PTB: {
            "lims": {"train": [10**0, 10**4.5], "valid": [10**1, 10**4.5]}
        },
        expdef.WT2: {
            "lims": {"train": [10**-1, 10**5], "valid": [10**1, 10**5]}
        },
    }

    def get_ylims(metric, dataset):
        if "accuracy" in metric or "f1" in metric:
            return [0, 105]
        elif metric == "training_loss":
            return ylims_trainingloss[dataset]["lims"]
        elif metric == "train_ppl":
            return ylims_PPL[dataset]["lims"]["train"]
        elif metric == "valid_ppl":
            return ylims_PPL[dataset]["lims"]["valid"]
        else:
            print(metric)
            raise ValueError

    for i, bs in enumerate(bss):
        for j, metric in enumerate(metrics_for_ds):
            ax = axes[j][i]
            ax.set_xscale("log")
            ax.set_ylim(get_ylims(metric, dataset))
            if plth.should_log(metric):
                ax.set_yscale("log", base=10)
                ax.tick_params(
                    axis="both", which="major", labelsize=plth.fontsizes["tiny"], pad=0
                )
                ax.yaxis.set_major_locator(ticker.LogLocator(numticks=5))
                ax.yaxis.set_minor_locator(ticker.LogLocator(numticks=5))
            else:
                ax.yaxis.set_major_locator(ticker.MultipleLocator(base=50))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=50))

    for ax in axes[-1]:
        ax.set_xlabel("Step-size", fontsize=plth.fontsizes["small"])
    for bs, ax in zip(bss, axes[0]):
        ax.set_title(plth.fdisplaynames(bs))
    for j, metric in enumerate(metrics_for_ds):
        axes[j][0].set_ylabel(plth.fdisplaynames(metric))

    plth.same_xlims(*plth.flatten(axes))


if __name__ == "__main__":
    settings(plt)

    data = load_data()

    for dataset in expdef.ALL_DS:
        for opt_to_plot in ["standard", "normalized"]:

            fig = plt.figure()
            make_figure(fig, data, dataset=dataset, opts_to_plot=opt_to_plot)
            filename = Path(__file__).stem + f"_{dataset}" + f"_{opt_to_plot}"
            plth.save(fig, name=os.path.join("output", filename))
            plt.close(fig)
