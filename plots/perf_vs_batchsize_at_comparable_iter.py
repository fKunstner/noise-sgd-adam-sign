import cmd
import os
from pathlib import Path

import explib.results.data as data_h
import explib.results.data_caching as data_cache
import explib.results.experiment_groups as expdef
import explib.results.plotting as plth
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from explib.results.cleanup import clean_data


def load_data():
    import importlib

    importlib.reload(data_cache)
    importlib.reload(data_h)
    importlib.reload(expdef)
    importlib.reload(data_cache)

    # runs_at_last_epoch, best_runs = data_cache.gridsearch_all_start()
    runs_at_last_epoch, best_runs = data_cache.gridsearch_all_start_soft_increase()
    data = [runs_at_last_epoch, best_runs]
    for i in range(len(data)):
        data[i] = data_h.add_stop_at_info(data[i], stop_at=expdef.EPOCH_CLIP_START_NEW)
    return data


def postprocess(data):
    return data


def settings(plt):
    plt.rcParams.update(
        plth.iclr_config_2(nrows=1, ncols=5, height_to_width_ratio=1 / 1.0)
    )
    pass


def make_figure(fig, data, opts_to_plot="normalized"):
    import importlib

    importlib.reload(plth)
    dss = [expdef.MNIST, expdef.CIFAR10, expdef.PTB, expdef.WT2, expdef.SQUAD]
    axes = plth.make_grid_iclr(fig, grid_type="2-3")
    best_runs = data[1]
    best_runs_stoped_epoch = best_runs[best_runs["epoch"] == best_runs["epoch_to_stop"]]

    if opts_to_plot == "standard":
        opts = expdef.STANDARD_OPT
    elif opts_to_plot == "normalized":
        opts = expdef.NORMALIZED_OPT
    else:
        raise ValueError(f"Opts to plot undef. Got {opts_to_plot}")

    YLIMS = {ds: (plth.MIN_LOSSES[ds], plth.INIT_LOSSES[ds]) for ds in expdef.ALL_DS}
    YLIMS.update(
        {
            # TODO
        }
    )

    for i, ds in enumerate(dss):
        res_ds = data_h.new_select(best_runs_stoped_epoch, selections=[{"dataset": ds}])
        for opt in opts:
            res_ds_bs_opt = data_h.new_select(res_ds, selections=[expdef.OPTIMS[opt]])
            res_ds_bs_opt = res_ds_bs_opt[res_ds_bs_opt["epoch"].notna()]
            agg = res_ds_bs_opt.groupby("eff_bs")["training_loss"].agg(
                [min, max, "median"]
            )

            bss, medians, mins, maxs = agg.index, agg["median"], agg["min"], agg["max"]
            # if ds == expdef.WT2:
            #    bss = bss[1:]
            # elif ds == expdef.SQUAD:
            #    bss = bss[2:]
            # medians, mins, maxs = medians[bss], mins[bss], maxs[bss]

            axes[0][i].plot(bss, medians, **plth.linestyles[opt])
            axes[0][i].fill_between(bss, mins, maxs, **plth.fillstyles[opt])

    for i, ds in enumerate(dss):
        ax = axes[0][i]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(plth.compute_limits(*plth.get_min_max(ax, axis="y"), margin_p=0.1))

        ylims_trainingloss = {
            expdef.MNIST: {
                "lims": [10**-5, 10**-0.5],
                "ticks": [-5, -3, -1],
            },
            expdef.CIFAR10: {
                "lims": [10**-6, 10**1],
                "ticks": [-6, -3, 0],
            },
            expdef.PTB: {"lims": [1.5, 6], "ticks": [2, 4, 6]},
            expdef.WT2: {"lims": [10**-1, 10**1.0], "ticks": [-1, 0, 1.0]},
            expdef.SQUAD: {"lims": [10**-1.25, 10**0.5], "ticks": [-1, 0]},
        }
        ax.set_ylim(ylims_trainingloss[ds]["lims"])
        if ds == expdef.PTB:
            ax.set_yticks(ylims_trainingloss[ds]["ticks"])
            ax.set_yticklabels(ylims_trainingloss[ds]["ticks"])
            ax.set_yticklabels([], minor=True)
        else:
            ax.set_yticks([10**i for i in ylims_trainingloss[ds]["ticks"]])
        # ax.set_yticks([], minor=True)
    axes[0][0].set_ylabel("Training loss \n at comparable iter")

    for i, ds in enumerate(dss):
        ax = axes[0][i]
        ax.set_title(plth.fdisplaynames(ds))
        xmin, xmax = plth.get_min_max(axes[0][i], axis="x")
        ax.set_xlim([xmin / 2, xmax * 2])
        xlabel = "Batch size"
        if ds == expdef.MNIST:
            plth.make_xticks_pow10(ax, [10**2, 10**3, 10**4, 10**5])
            xlabel = "" + xlabel + ""
        if ds == expdef.CIFAR10:
            plth.make_xticks_pow10(ax, [10**2, 10**3, 10**4, 10**5])
            xlabel = "    " + xlabel + ""
        if ds == expdef.PTB:
            plth.make_xticks_pow10(ax, [10**1, 10**2, 10**3, 10**4, 10**5])
            xlabel = "" + xlabel + ""
        if ds == expdef.WT2:
            plth.make_xticks_pow10(ax, [10**1, 10**2, 10**3, 10**4])
            xlabel = "" + xlabel + "   "
        if ds == expdef.SQUAD:
            plth.make_xticks_pow10(ax, [10**1, 10**2, 10**3, 10**4, 10**5])
            xlabel = "" + xlabel + ""
        ax.set_xlabel(xlabel, labelpad=-4, fontsize="x-small")


if __name__ == "__main__":
    settings(plt)
    data = postprocess(load_data())

    fig = plt.figure()
    make_figure(fig, data, opts_to_plot="standard")
    filename = Path(__file__).stem
    plth.save(fig, name=os.path.join("output", filename))
    plt.close(fig)

    fig = plt.figure()
    make_figure(fig, data, opts_to_plot="normalized")
    filename = Path(__file__).stem
    plth.save(fig, name=os.path.join("output", filename + "_norm"))
    plt.close(fig)
