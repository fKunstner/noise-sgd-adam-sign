import importlib
import os
from pathlib import Path

import explib.results.data as data_h
import explib.results.experiment_groups as expdef
import explib.results.plotting as plth
import matplotlib
import matplotlib.pyplot as plt
from explib.results import data_caching


def load_data():
    runs_at_last_epoch, best_runs = data_caching.gridsearch_nd_all_end()
    return {"best_runs": best_runs}


def settings(plt):
    importlib.reload(plth)

    plt.rcParams.update(
        plth.iclr_config_2(
            rel_width=1.0, nrows=2, ncols=2, height_to_width_ratio=1 / 2.0
        )
    )


def make_figure(fig, data):
    importlib.reload(plth)
    importlib.reload(expdef)

    best_runs = data["best_runs"]

    axes = plth.make_grid_iclr(fig, grid_type="2x2")
    optss = [
        [expdef.SGD_M, expdef.ADAM_M, expdef.SIGN_M, expdef.NORM_M],
        [expdef.SGD_NM, expdef.ADAM_NM, expdef.SIGN_NM, expdef.NORM_NM],
    ]
    dss = [expdef.PTB, expdef.WT2]

    for i, ds in enumerate(dss):
        results = data_h.new_select(
            best_runs, selections=expdef.SPECIAL[expdef.NO_DROPOUT][ds][expdef.FULL]
        )
        results = results[results["epoch"].notna()]
        results = results[results["epoch"] <= expdef.EPOCH_CLIP[ds][expdef.FULL]]
        for j, opts in enumerate(optss):
            ax = axes[j][i]
            for opt in opts:
                res_opt = data_h.new_select(results, selections=[expdef.OPTIMS[opt]])
                agg = res_opt.groupby("epoch")["training_loss"].agg(
                    [min, max, "median"]
                )

                n_samples = 3200
                ax.plot(
                    plth.subsample(agg.index, n_samples),
                    plth.subsample(agg["median"], n_samples),
                    **plth.linestyles_nm[opt],
                )
                fillstyle = plth.fillstyles[opt]
                ax.fill_between(
                    plth.subsample(agg.index, n_samples),
                    plth.subsample(agg["min"], n_samples),
                    plth.subsample(agg["max"], n_samples),
                    **fillstyle,
                )

    ylims_trainingloss = {
        expdef.PTB: [10**-1.0, 10],
        expdef.WT2: [10**-1.0, 10**1.3],
    }

    for i, ds in enumerate(dss):
        for ax in [axes[0][i], axes[1][i]]:
            ax.set_ylim(ylims_trainingloss[ds])
            ax.set_yscale("log")

        axes[0][i].set_title(plth.fdisplaynames(ds))
        axes[1][i].set_xlabel("Epoch")
    axes[0][0].set_ylabel("Training loss")
    axes[1][0].set_ylabel("Training loss")


if __name__ == "__main__":
    settings(plt)
    fig = plt.figure()
    make_figure(fig, load_data())
    plth.save(fig, name=os.path.join("output", Path(__file__).stem))
