import importlib
import os
from pathlib import Path

import explib.results.data as data_h
import explib.results.experiment_groups as expdef
import explib.results.plotting as helpers
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from explib.results import data_caching
from matplotlib import gridspec
from statsmodels.graphics.gofplots import qqplot

plth = helpers


def load_data(seed=4):
    gradnorms = plth.load_gradnorms(seed)
    runs_at_last_epoch, best_runs = data_caching.gridsearch_all_end()
    best_runs = data_h.add_stop_at_info(best_runs, stop_at=expdef.EPOCH_CLIP)
    return {"optims": best_runs, "gradnorms": gradnorms}


def settings(plt):
    plt.rcParams.update(
        plth.iclr_config_2(
            rel_width=1.0, nrows=2, ncols=5, height_to_width_ratio=1 / 1.0
        )
    )


def make_figure(fig, data):
    importlib.reload(helpers)
    importlib.reload(expdef)

    data_optims = data["optims"]
    data_gradnorms = data["gradnorms"]

    dsets = ["mnist", "cifar", "ptb", "wt2", "squad"]
    settings = {
        "mnist": {
            "title": plth.fdisplaynames("mnist"),
            "norm_name": "mnist_256",
        },
        "cifar": {
            "title": plth.fdisplaynames("cifar10"),
            "norm_name": "cifar10_64",
        },
        "ptb": {
            "title": plth.fdisplaynames("ptb"),
            "norm_name": "ptb_16",
        },
        "wt2": {
            "title": plth.fdisplaynames("wikitext2"),
            "norm_name": "wt2_16",
        },
        "squad": {
            "title": plth.fdisplaynames("squad"),
            "norm_name": "squad_16",
        },
    }
    axes = plth.make_grid_iclr(fig, grid_type="2x2-3")

    zoomax_bottom = 0.2
    zoomax_left = 0.38 + 0.12
    zoomax_width = 0.59 - 0.12
    zoomaxes = [
        axes[0][2].inset_axes(bounds=(zoomax_left, zoomax_bottom, zoomax_width, 0.35)),
        axes[0][3].inset_axes(bounds=(zoomax_left, zoomax_bottom, zoomax_width, 0.35)),
        axes[0][4].inset_axes(bounds=(zoomax_left, zoomax_bottom, zoomax_width, 0.35)),
    ]
    plot_norm_squared = False
    transform = (lambda x: x) if plot_norm_squared else np.sqrt
    zoom_settings = {
        "ptb": {
            "id": 0,
            "ymax": 14,
            "xmin": transform(0.575),
            "xmax": transform(0.989),
        },
        "wt2": {"id": 1, "ymax": 14, "xmin": transform(1.115), "xmax": transform(1.6)},
        "squad": {
            "id": 2,
            "ymax": 14,
            "xmin": transform(19.9),
            "xmax": transform(48.6),
        },
    }

    plth.hide_frame(*zoomaxes, left=True)

    qq_w = 0.425
    qq_h = 0.475
    qqaxes = [ax.inset_axes(bounds=(1 - qq_w, 1 - qq_h, qq_w, qq_h)) for ax in axes[0]]

    for ax in qqaxes:
        for name, spine in ax.spines.items():
            spine.set_linewidth(0.6)

    helpers.hide_frame(*qqaxes, top=False, right=False, left=False, bottom=False)

    helpers.hide_ticks(*qqaxes)

    helpers.hide_frame(*zoomaxes, top=True, right=True, left=True, bottom=False)
    helpers.hide_ticks(*zoomaxes, x=True, y=False)

    C = [0.3333333333] * 3
    colors = {
        "mnist": C,
        "cifar": C,
        "ptb": C,
        "wt2": C,
        "squad": C,
    }

    for i, dset in enumerate(dsets):
        xs = data_gradnorms[settings[dset]["norm_name"]][:-1]
        axes[0][i].hist(transform(xs), bins=50, color=colors[dset])
        ax = axes[0][i]
        ax.set_title(settings[dset]["title"])

        if True:
            if dset in zoom_settings.keys():
                zoom_setting = zoom_settings[dset]

                zoomax = zoomaxes[zoom_setting["id"]]
                zoomax.hist(transform(xs), bins=50, color=colors[dset])
                zoomax.set_xlim([zoom_setting["xmin"], zoom_setting["xmax"]])
                zoomax.set_ylim([0, zoom_setting["ymax"]])

                left_in_datacoords = (
                    zoomax_left * (ax.get_xlim()[1] - ax.get_xlim()[0])
                    + ax.get_xlim()[0]
                )
                right_in_datacoords = (zoomax_left + zoomax_width) * (
                    ax.get_xlim()[1] - ax.get_xlim()[0]
                ) + ax.get_xlim()[0]

                ax.plot(
                    [zoom_setting["xmin"], left_in_datacoords],
                    [0, zoomax_bottom * ax.get_ylim()[1]],
                    linewidth=helpers._stroke_width,
                    color=helpers.BASE_COLORS["gray"],
                )
                ax.plot(
                    [ax.get_xlim()[1], right_in_datacoords],
                    [0, zoomax_bottom * ax.get_ylim()[1]],
                    linewidth=helpers._stroke_width,
                    color=helpers.BASE_COLORS["gray"],
                )

                ax.set_xlim([ax.get_xlim()[0], ax.get_xlim()[1] + 0.001])

                zoomax.tick_params(
                    axis="both", which="major", labelsize=helpers.fontsizes["tiny"]
                )

    axes[0][0].set_ylabel("Counts")
    for ax in axes[0]:
        ax.set_xlabel("Gradient error")
        ax.grid(False)

    for i, dset in enumerate(dsets):
        xs = data_gradnorms[settings[dset]["norm_name"]][:-1]
        ax = qqaxes[i]
        qqplot(
            data=transform(xs),
            ax=ax,
            fit=True,
            line=None,
            markersize=0.5,
            marker=".",
            markeredgecolor=colors[dset],
            linestyle="none",
        )

        ax.set_xlabel("", fontsize=helpers.fontsizes["tiny"])
        ax.set_ylabel("", fontsize=helpers.fontsizes["tiny"])

        end_pts = list(zip(ax.get_xlim(), ax.get_ylim()))
        end_pts[0] = min(end_pts[0])
        end_pts[1] = max(end_pts[1])
        ax.plot(
            end_pts,
            end_pts,
            color=helpers.BASE_COLORS["gray"],
            zorder=0,
            linewidth=helpers._stroke_width,
        )

        ax.set_xlim(end_pts)
        ax.set_ylim(end_pts)

    dss = [expdef.MNIST, expdef.CIFAR10, expdef.PTB, expdef.WT2, expdef.SQUAD]
    for i, ds in enumerate(dss):
        ax = axes[1][i]
        res_ds = data_h.new_select(
            data_optims, selections=expdef.EXPERIMENTS[ds][expdef.S]
        )
        epoch_clip = expdef.EPOCH_CLIP[ds][expdef.S]
        res_ds = res_ds[res_ds["epoch"] <= epoch_clip]

        for opt in expdef.STANDARD_OPT:
            res_ds_bs_opt = data_h.new_select(res_ds, selections=[expdef.OPTIMS[opt]])
            res_ds_bs_opt = res_ds_bs_opt[res_ds_bs_opt["epoch"].notna()]

            agg = res_ds_bs_opt.groupby("step")["training_loss"].agg(
                [min, max, "median"]
            )
            n_samples = 50
            ax.plot(
                plth.subsample(agg.index, n_samples),
                plth.subsample(agg["median"], n_samples),
                **plth.linestyles_nm[opt],
            )
            ax.fill_between(
                plth.subsample(agg.index, n_samples),
                plth.subsample(agg["min"], n_samples),
                plth.subsample(agg["max"], n_samples),
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

    for i, ds in enumerate(expdef.ALL_DS):
        ax = axes[1][i]
        if ds == expdef.MNIST:
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xticklabels([0, "", "", "", 100])
        if ds == expdef.CIFAR10:
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xticklabels([0, "", "", "", 100])
        if ds == expdef.PTB:
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xticklabels([0, "", "", "", 100])
        if ds == expdef.WT2:
            ax.set_xticks([0, 10, 20, 30, 40])
            ax.set_xticklabels([0, "", "", "", 40])
        if ds == expdef.SQUAD:
            ax.set_xticks([0, 1, 2, 3, 4, 5])
            ax.set_xticklabels([0, "", "", "", "", 5])
        ax.set_xlabel("Epoch", labelpad=-5)
    axes[1][0].set_ylabel("Training loss")

    make_legend = False
    if make_legend:
        legsettings = {
            "frameon": False,
            "borderaxespad": -0.3,
            "labelspacing": 0.1,
            "handlelength": 1.3,
            "handletextpad": 0.3,
            "fontsize": "x-small",
            "markerfirst": False,
        }
        linestyles = [
            plth.linestyles_nm[expdef.SGD_M],
            plth.linestyles_nm[expdef.SGD_NM],
            plth.linestyles_nm[expdef.ADAM_M],
            plth.linestyles_nm[expdef.ADAM_NM],
        ]
        for i in range(len(linestyles)):
            linestyles[i]["linewidth"] = 1.5
        linestyles[1]["dashes"] = (2.0, 2.0)
        linestyles[3]["dashes"] = (2.0, 2.0)
        lines = [
            matplotlib.lines.Line2D([0, 1], [0, 1], **linestyle)
            for linestyle in linestyles
        ]
        labels = [
            plth.fdisplaynames(expdef.SGD_M),
            plth.fdisplaynames(expdef.SGD_NM),
            plth.fdisplaynames(expdef.ADAM_M),
            plth.fdisplaynames(expdef.ADAM_NM),
        ]

        lines_labels_and_ax = [
            ([lines[1]], [labels[1]], axes[1][0]),
            ([lines[3]], [labels[3]], axes[1][1]),
            ([lines[0]], [labels[0]], axes[1][2]),
            ([lines[2]], [labels[2]], axes[1][3]),
        ]
        lines_labels_and_ax = [
            ([lines[1], lines[3]], [labels[1], labels[3]], axes[1][2]),
            ([lines[0], lines[2]], [labels[0], labels[2]], axes[1][4]),
        ]
        for lines, labels, ax in lines_labels_and_ax:
            legend = ax.legend(lines, labels, **legsettings, loc="upper right")

    fig.canvas.draw()


if __name__ == "__main__":
    settings(plt)
    fig = plt.figure()
    make_figure(fig, load_data())
    helpers.save(fig, name=os.path.join("output", Path(__file__).stem))
