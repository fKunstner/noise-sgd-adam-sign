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
    return {"best_runs": best_runs, "last_epoch": runs_at_last_epoch}


def settings(plt):
    importlib.reload(datah)
    importlib.reload(cleanh)
    importlib.reload(plth)

    plt.rcParams.update(plth.iclr_config_2(nrows=3, ncols=5, height_to_width_ratio=1))


def make_figure(fig, data, dataset=expdef.WT2, opts_to_plot="normalized"):
    importlib.reload(datah)
    importlib.reload(cleanh)
    importlib.reload(plth)
    importlib.reload(expdef)

    fig.set_dpi(250)

    best_runs = data["best_runs"]

    if opts_to_plot == "standard":
        opts = [expdef.SGD_NM, expdef.ADAM_NM, expdef.SGD_M, expdef.ADAM_M]
    elif opts_to_plot == "normalized":
        opts = [expdef.SIGN_NM, expdef.NORM_NM, expdef.SIGN_M, expdef.NORM_M]
    else:
        raise ValueError(f"Unknown opts {opts_to_plot}")

    bss = expdef.ALL_BS
    grid_type = "3x5"
    metrics_for_ds = [
        plth.metric_type_to_dset_to_metric[metric][dataset]
        for metric in ["training_loss", "training_perf", "validation_perf"]
    ]
    axes = plth.make_grid_iclr(fig, grid_type=grid_type, tight=True)

    for i, bs in enumerate(bss):
        best_runs_bs = data_h.new_select(best_runs, expdef.EXPERIMENTS[dataset][bs])
        best_runs_bs = best_runs_bs[best_runs_bs["epoch"].notna()]
        best_runs_bs = best_runs_bs[
            best_runs_bs["epoch"] <= expdef.EPOCH_CLIP[dataset][bs]
        ]
        for opt in opts:
            best_runs_ = data_h.new_select(best_runs_bs, [expdef.OPTIMS[opt]])
            agg = best_runs_.groupby("epoch")[metrics_for_ds].agg([min, max, "median"])
            for j, metric in enumerate(metrics_for_ds):
                ax = axes[j][i]

                n_subsample = 100
                linestyle = plth.linestyles_nm[opt]
                ax.plot(
                    plth.subsample(agg.index, n=n_subsample),
                    plth.subsample(agg[metric]["median"], n=n_subsample),
                    **linestyle,
                )
                ax.fill_between(
                    plth.subsample(agg.index, n=n_subsample),
                    plth.subsample(agg[metric]["min"], n=n_subsample),
                    plth.subsample(agg[metric]["max"], n=n_subsample),
                    **plth.fillstyles[opt],
                )

    ylims_trainingloss = {
        expdef.MNIST: {
            "lims": [10**-6, 10**1],
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
        ax.set_xlabel("Epoch", fontsize=plth.fontsizes["small"])
    for bs, ax in zip(bss, axes[0]):
        ax.set_title(plth.fdisplaynames(bs))
    for j, metric in enumerate(metrics_for_ds):
        axes[j][0].set_ylabel(plth.fdisplaynames(metric))


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
