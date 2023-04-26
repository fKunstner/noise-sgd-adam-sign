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
import matplotlib.colors
import matplotlib.pyplot as plt


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

    plot_image = False

    if plot_image:
        plt.rcParams.update(
            plth.iclr_config_2(nrows=2, ncols=5, height_to_width_ratio=1)
        )
    else:
        plt.rcParams.update(
            plth.iclr_config_2(nrows=2, ncols=5, height_to_width_ratio=1 / 1.2)
        )


def make_figure(fig, data, with_momentum=False):
    importlib.reload(datah)
    importlib.reload(cleanh)
    importlib.reload(plth)
    importlib.reload(expdef)

    dss = [expdef.PTB, expdef.WT2, expdef.SQUAD]
    bss = [expdef.M, expdef.FULL]
    if with_momentum:
        opts = [expdef.SGD_M, expdef.ADAM_M, expdef.SIGN_M, expdef.NORM_M]
    else:
        opts = [expdef.SGD_NM, expdef.ADAM_NM, expdef.SIGN_NM, expdef.NORM_NM]

    grid_type = "2x3"
    plot_data = data["plot_data"]

    def quickselect_agg(ds, bs, opt):
        res_ds_bs = data_h.new_select(plot_data, selections=expdef.EXPERIMENTS[ds][bs])
        res_ds_bs = res_ds_bs[res_ds_bs["epoch"].notna()]
        res_ds_bs = res_ds_bs[res_ds_bs["epoch"] <= expdef.EPOCH_CLIP[ds][bs]]
        res_opt = data_h.new_select(res_ds_bs, selections=[expdef.OPTIMS[opt]])
        agg = res_opt.groupby("epoch")["training_loss"].agg([min, max, "median"])
        return agg

    axes = plth.make_grid_iclr(fig, grid_type=grid_type, tight=True)
    for i, bs in enumerate(bss):
        for j, ds in enumerate(dss):
            ax = axes[i][j]
            for opt in opts:
                agg = quickselect_agg(ds, bs, opt)
                n_samples = 100
                linestyle = plth.linestyles_nm[opt]
                linestyle["linewidth"] = 1.0 if "Sign" in opt else 0.9
                linestyle["linestyle"] = "-" if "Sign" in opt else "dotted"
                linestyle.pop("dashes", None)
                ax.plot(
                    plth.subsample(agg.index, n_samples),
                    plth.subsample(agg["median"], n_samples),
                    **linestyle,
                )
                fillstyle = plth.fillstyles[opt]
                fillstyle["alpha"] = 0.1
                ax.fill_between(
                    plth.subsample(agg.index, n_samples),
                    plth.subsample(agg["min"], n_samples),
                    plth.subsample(agg["max"], n_samples),
                    **fillstyle,
                )
            ax.set_yscale("log")

    ylims_trainingloss = {
        expdef.MNIST: [10**-6, 10**1],
        expdef.CIFAR10: [10**-7, 10**1.5],
        expdef.PTB: [1.7, 10],
        expdef.WT2: [10**-1.0, 10**1.3],
        expdef.SQUAD: [10**-1, 10**1.0],
    }

    for i, bs in enumerate(bss):
        for j, ds in enumerate(dss):
            ax = axes[i][j]
            ax.set_ylim(ylims_trainingloss[ds])
            if ds == expdef.PTB:
                ax.set_yticks([2, 4, 8], minor=False)
                ax.set_yticklabels([2, 4, 8], minor=False)
                ax.set_yticks([2, 3, 4, 5, 6, 7, 8, 9, 10], minor=True)
                ax.set_yticklabels([], minor=True)

    xticks_and_lims = {
        expdef.PTB: {"ticks": [0, 1000, 2000, 3000], "labels": [0, "", "", 3000]},
        expdef.WT2: {"ticks": [0, 100, 200, 300], "labels": [0, "", "", 300]},
        expdef.SQUAD: {"ticks": [0, 20, 40, 60], "labels": [0, "", "", 60]},
    }

    for j, ds in enumerate(dss):
        axes[0][j].set_title(plth.fdisplaynames(ds), y=1.0, pad=-1)
        axes[1][j].set_xticks(xticks_and_lims[ds]["ticks"])
        axes[1][j].set_xticklabels(xticks_and_lims[ds]["labels"])
        axes[1][j].set_xlabel("Epoch            ", labelpad=-7)

    axes[0][0].set_ylabel("Medium batch\nTraining Loss")
    axes[1][0].set_ylabel("Full batch\nTraining Loss")

    ## Names

    def darker(color):
        black = [0.0, 0.0, 0.0]
        black_hsv = matplotlib.colors.rgb_to_hsv(black)
        color_hsv = matplotlib.colors.rgb_to_hsv(color)
        a = 0.0
        avg_hsv = [a * bi + (1 - a) * ci for (bi, ci) in zip(black_hsv, color_hsv)]
        avg = matplotlib.colors.hsv_to_rgb(avg_hsv)
        a = 0.35
        avg = [a * bi + (1 - a) * ci for (bi, ci) in zip(black, color)]
        return avg

    def no_mom(text):
        return text.replace("($+$m)", "").replace("($-$m)", "")

    dy = {
        expdef.M: {
            expdef.PTB: {
                expdef.SIGN_M: 0.6,
                expdef.SGD_M: 0.33,
                expdef.NORM_M: 0.23,
                expdef.ADAM_M: 0.10,
                expdef.SIGN_NM: 0.6,
                expdef.SGD_NM: 0.33,
                expdef.NORM_NM: 0.23,
                expdef.ADAM_NM: 0.10,
            },
            expdef.WT2: {
                expdef.SIGN_M: 0.6,
                expdef.SGD_M: 0.7,
                expdef.NORM_M: 0.37,
                expdef.ADAM_M: 0.05,
                expdef.SIGN_NM: 0.6,
                expdef.SGD_NM: 0.7,
                expdef.NORM_NM: 0.45,
                expdef.ADAM_NM: 0.15,
            },
            expdef.SQUAD: {
                expdef.SGD_M: 0.35,
                expdef.NORM_M: 0.25,
                expdef.SIGN_M: 0.12,
                expdef.ADAM_M: 0.02,
                expdef.SGD_NM: 0.35,
                expdef.NORM_NM: 0.25,
                expdef.SIGN_NM: 0.12,
                expdef.ADAM_NM: 0.02,
            },
        },
        expdef.FULL: {
            expdef.PTB: {
                expdef.SGD_M: 0.45,
                expdef.NORM_M: 0.25,
                expdef.SIGN_M: 0.13,
                expdef.ADAM_M: 0.02,
                expdef.SGD_NM: 0.55,
                expdef.NORM_NM: 0.43,
                expdef.SIGN_NM: 0.18,
                expdef.ADAM_NM: 0.06,
            },
            expdef.WT2: {
                expdef.SGD_M: 0.77,
                expdef.NORM_M: 0.67,
                expdef.SIGN_M: 0.39,
                expdef.ADAM_M: 0.05,
                expdef.SGD_NM: 0.90,
                expdef.NORM_NM: 0.80,
                expdef.SIGN_NM: 0.65,
                expdef.ADAM_NM: 0.53,
            },
            expdef.SQUAD: {
                expdef.SGD_M: 0.50,
                expdef.NORM_M: 0.40,
                expdef.ADAM_M: 0.12,
                expdef.SIGN_M: 0.02,
                expdef.SGD_NM: 0.65,
                expdef.NORM_NM: 0.54,
                expdef.SIGN_NM: 0.42,
                expdef.ADAM_NM: 0.30,
            },
        },
    }

    for i, bs in enumerate(bss):
        for j, ds in enumerate(dss):
            ax = axes[i][j]
            xlims = ax.get_xlim()
            ax.set_xlim([xlims[0], 1.2 * xlims[1]])
            ylims = ax.get_ylim()

            for opt in opts:
                agg = quickselect_agg(ds, bs, opt)
                x = agg.index[-1] + (xlims[1] - xlims[0]) * 0.025
                y = (ylims[0] ** (1 - dy[bs][ds][opt])) * (
                    ylims[1] ** (dy[bs][ds][opt])
                )

                color = darker(plth.linestyles_nm[opt]["color"])

                ax.text(
                    x, y, no_mom(plth.abbrev(opt)), fontsize="xx-small", color=color
                )


if __name__ == "__main__":
    settings(plt)

    data = load_data()

    for with_momentum in [True, False]:
        fig = plt.figure()
        make_figure(fig, data, with_momentum=with_momentum)
        filename = Path(__file__).stem
        if not with_momentum:
            filename += "_nomom"
        plth.save(fig, name=os.path.join("output", filename))
        plt.close(fig)
