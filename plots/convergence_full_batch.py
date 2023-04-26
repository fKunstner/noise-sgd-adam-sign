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
    runs_at_last_epoch, best_runs = data_caching.gridsearch_all_end()
    return {"optims": best_runs}


def settings(plt):
    importlib.reload(plth)

    plt.rcParams.update(
        plth.iclr_config_2(
            rel_width=1.0, nrows=1, ncols=5, height_to_width_ratio=1 / 1.0
        )
    )


def make_figure(fig, data):
    importlib.reload(plth)

    data_optims = data["optims"]
    axes = plth.make_grid_iclr(fig, grid_type="2-3")

    for i, ds in enumerate(expdef.ALL_DS):
        ax = axes[0][i]
        res_ds = data_h.new_select(
            data_optims, selections=expdef.EXPERIMENTS[ds][expdef.FULL]
        )
        for opt in expdef.STANDARD_OPT:
            res_ds_opt = data_h.new_select(res_ds, selections=[expdef.OPTIMS[opt]])
            res_ds_opt = res_ds_opt[res_ds_opt["epoch"].notna()]

            agg = res_ds_opt.groupby("step")[["training_loss", "epoch"]].agg(
                [min, max, "median"]
            )

            epochs_min, epochs_max = min(agg[("epoch", "median")]), max(
                agg[("epoch", "median")]
            )
            n_points = 50
            ax.plot(
                plth.subsample(agg[("epoch", "median")], n_points),
                plth.subsample(agg[("training_loss", "median")], n_points),
                **plth.linestyles_nm[opt],
            )
            ax.fill_between(
                plth.subsample(agg[("epoch", "median")], n_points),
                plth.subsample(agg[("training_loss", "min")], n_points),
                plth.subsample(agg[("training_loss", "max")], n_points),
                **plth.fillstyles[opt],
            )

        ax.set_yscale("log")

        if ds == expdef.MNIST:
            ax.set_ylim([10**-6, 10**1])

        if ds == expdef.CIFAR10:
            ax.set_ylim([10**-5, 10**2])
        if ds == expdef.PTB:
            ax.set_ylim([1.7, 7])
            ax.set_yticks([], minor=False)
            ax.set_yticks([2, 3, 4, 5, 6, 7, 8, 9], minor=True)
            ax.set_yticklabels([2, "", 4, "", 6, "", 8, ""], minor=True)
        if ds == expdef.WT2:
            ax.set_ylim([10**-1, 10**1.5])
        if ds == expdef.SQUAD:
            ax.set_ylim([10**-1, 10**1])

        ax.set_title(plth.fdisplaynames(ds))

        ax.set_xticks([epochs_min, epochs_max / 2, epochs_max])
        ax.set_xticklabels([0, "", int(epochs_max)])
        ax.set_xlabel("Epoch", labelpad=-5)

    make_legend = False
    if make_legend:
        legsettings = {
            "frameon": False,
            "borderaxespad": -0.1,
            "labelspacing": 0.1,
            "handlelength": 1.8,
            "handletextpad": 0.3,
            "fontsize": "x-small",
            "markerfirst": False,
        }
        lines = axes[0][0].lines
        axes[0][3].legend(
            [lines[1], lines[3]],
            [plth.fdisplaynames("adam-m"), plth.fdisplaynames("sgd-m")],
            **legsettings,
            loc="best",
        )
        axes[0][0].legend(
            [lines[0], lines[2]],
            [plth.fdisplaynames("sgd+m"), plth.fdisplaynames("adam+m")],
            **legsettings,
            loc="best",
        )

    axes[0][0].set_ylabel("Training loss")
    fig.canvas.draw()


if __name__ == "__main__":
    settings(plt)
    fig = plt.figure()
    make_figure(fig, load_data())
    plth.save(fig, name=os.path.join("output", Path(__file__).stem))
