"""Attempt at a figure that would show."""


import cmd
import importlib
import os
from pathlib import Path

import explib.results.cleanup as cleanh
import explib.results.data as data_h
import explib.results.data as datah
import explib.results.data_caching as data_cache
import explib.results.experiment_groups as expdef
import explib.results.plotting as plth
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from explib.results.cleanup import clean_data
from tqdm import tqdm


def load_data():
    importlib.reload(datah)
    importlib.reload(cleanh)
    importlib.reload(plth)

    runs_at_last_epoch, best_runs = data_cache.gridsearch_all_end()

    return {"plot_data": best_runs}


def settings(plt):
    importlib.reload(datah)
    importlib.reload(cleanh)
    importlib.reload(plth)

    plt.rcParams.update(
        plth.iclr_config_2(nrows=2, ncols=5, height_to_width_ratio=1 / 1.2)
    )


def make_figure(fig, data, with_image=True, opts_to_plot="standard"):
    importlib.reload(datah)
    importlib.reload(cleanh)
    importlib.reload(plth)
    importlib.reload(expdef)

    if opts_to_plot == "standard":
        opts1 = [expdef.SGD_NM, expdef.ADAM_NM]
        opts2 = [expdef.SGD_M, expdef.ADAM_M]
    elif opts_to_plot == "normalized":
        opts1 = [expdef.SIGN_NM, expdef.NORM_NM]
        opts2 = [expdef.SIGN_M, expdef.NORM_M]
    else:
        raise ValueError(f"Unknown opts {opts_to_plot}")

    dss = expdef.ALL_DS if with_image else [expdef.PTB, expdef.WT2, expdef.SQUAD]
    grid_type = "2x2-3" if with_image else "2x3"
    plot_data = data["plot_data"]
    axes = plth.make_grid_iclr(fig, grid_type=grid_type, tight=True)

    def make_plottype(ax, plot_data, ds, bs, opts):
        res_ds_bs = data_h.new_select(plot_data, selections=expdef.EXPERIMENTS[ds][bs])
        res_ds_bs = res_ds_bs[res_ds_bs["epoch"].notna()]
        res_ds_bs = res_ds_bs[res_ds_bs["epoch"] <= expdef.EPOCH_CLIP[ds][bs]]

        res_ds_bs["iter"] = res_ds_bs["epoch"] * res_ds_bs["grad_updates_per_epoch"] + 1

        for opt in opts:
            res_opt = data_h.new_select(res_ds_bs, selections=[expdef.OPTIMS[opt]])
            agg = res_opt.groupby("iter")["training_loss"].agg([min, max, "median"])
            n_samples = 50
            ax.plot(
                plth.subsample(agg.index, n_samples),
                plth.subsample(agg["median"], n_samples),
                **plth.linestyles_nm[opt],
                alpha=0.4,
            )
            fillstyle = plth.fillstyles[opt]
            fillstyle["alpha"] = 0.1
            ax.fill_between(
                plth.subsample(agg.index, n_samples),
                plth.subsample(agg["min"], n_samples),
                plth.subsample(agg["max"], n_samples),
                **fillstyle,
            )
            linestyle_ = plth.linestyles[opt]
            markersize = {
                bs: x
                for (bs, x) in zip(
                    expdef.ALL_BS, np.sqrt(np.linspace(1**2, 4.0**2, 5))
                )
            }

            linestyle_["markersize"] = markersize[bs]
            linestyle_["marker"] = "v"
            ax.plot(
                list(agg.index)[-1],
                list(agg["median"])[-1],
                **linestyle_,
            )

    for line in [0, 1]:
        for j, ds in enumerate(dss):
            ax = axes[line][j]
            for idx, bs in enumerate(expdef.ALL_BS):
                if line == 0:
                    make_plottype(ax, plot_data, ds, bs, opts=opts1)
                else:
                    make_plottype(ax, plot_data, ds, bs, opts=opts2)

            ax.set_xscale("log", base=10)
            ax.set_xlim([1, ax.get_xlim()[1] * 2])

            ylims_trainingloss = {
                expdef.MNIST: [10**-6, 10**1],
                expdef.CIFAR10: [10**-7, 10**1.5],
                expdef.PTB: [1.7, 10],
                expdef.WT2: [10**-1.0, 10**1.3],
                expdef.SQUAD: [10**-1, 10**1.0],
            }
            ax.set_ylim(ylims_trainingloss[ds])
            ax.set_yscale("log")
            if ds == expdef.PTB:
                ax.set_yticks([2, 4, 8], minor=False)
                ax.set_yticklabels([2, 4, 8], minor=False)
                ax.set_yticks([2, 3, 4, 5, 6, 7, 8, 9, 10], minor=True)
                ax.set_yticklabels([], minor=True)

    xticks_and_lims = {
        expdef.MNIST: {"lim": [10**1, 10**4.5], "ticks": [1, 2, 3, 4, 5]},
        expdef.CIFAR10: {"lim": [10**1, 10**5], "ticks": [1, 2, 3, 4, 5]},
        expdef.PTB: {"lim": [10**1, 10**6], "ticks": [1, 2, 3, 4, 5, 6]},
        expdef.WT2: {"lim": [10**1, 10**5], "ticks": [1, 2, 3, 4, 5]},
        expdef.SQUAD: {"lim": [10**0.5, 10**5], "ticks": [1, 2, 3, 4, 5]},
    }

    for i, ds in enumerate(dss):
        ticks = [10**i for i in xticks_and_lims[ds]["ticks"]]

        axes[0][i].set_xlim(xticks_and_lims[ds]["lim"])
        axes[1][i].set_xlim(xticks_and_lims[ds]["lim"])

        axes[0][i].set_xticks(ticks, minor=False)
        plth.make_xticks_pow10(axes[1][i], ticks)

    for ax, ds in zip(axes[0], dss):
        ax.set_title(
            f"{plth.fdisplaynames(ds)}",
            y=1.0,
            pad=-0.0,
        )

    for ax in axes[1]:
        ax.set_xlabel("Iteration", labelpad=-7)

    axes[0][0].set_ylabel("Training loss")
    axes[1][0].set_ylabel("Training loss")


if __name__ == "__main__":
    settings(plt)

    data = load_data()

    for opts_to_plot in ["standard", "normalized"]:
        for with_image in [True, False]:

            fig = plt.figure()
            make_figure(fig, data, with_image=with_image, opts_to_plot=opts_to_plot)

            filename = Path(__file__).stem
            if opts_to_plot == "normalized":
                filename += "_norm"
            if with_image:
                filename += "_with_image"

            plth.save(fig, name=os.path.join("output", filename))
            plt.close(fig)
