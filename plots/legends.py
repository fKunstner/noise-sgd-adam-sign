"""Attempt at a figure that would show."""


import importlib
import os
from pathlib import Path

import explib.results.cleanup as cleanh
import explib.results.data as datah
import explib.results.experiment_groups as expdef
import explib.results.plotting as plth
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def load_data():
    return {}


def settings(plt):
    importlib.reload(datah)
    importlib.reload(cleanh)
    importlib.reload(plth)
    plt.rcParams.update(
        plth.icml_config(nrows=1, ncols=1, height_to_width_ratio=1 / 50)
    )


def make_figure(fig, data, normalized=True, with_bs=True):
    importlib.reload(datah)
    importlib.reload(cleanh)
    importlib.reload(plth)

    if normalized:
        optims = [expdef.SIGN_M, expdef.NORM_M, expdef.SIGN_NM, expdef.NORM_NM]
    else:
        optims = [expdef.SGD_M, expdef.ADAM_M, expdef.SGD_NM, expdef.ADAM_NM]

    lines = []
    for opt in optims:
        linestyle = plth.linestyles_nm[opt].copy()
        linestyle["linewidth"] = 1.75
        if "+m" not in opt:
            linestyle["dashes"] = (2.0, 2.0)

        label = plth.abbrev(opt)
        lines.append(matplotlib.lines.Line2D([], [], **linestyle, label=label))

    if with_bs:
        leg = fig.legend(
            handles=lines,
            loc="center left",
            ncol=len(optims),
            frameon=False,
            borderpad=1,
            fontsize="small",
            handletextpad=0.5,
            handlelength=1.5,
            columnspacing=1.25,
        )
        fig.add_artist(leg)
    else:
        leg = fig.legend(
            handles=lines,
            loc="center",
            ncol=len(optims),
            frameon=False,
            borderpad=0,
            fontsize="small",
            handletextpad=0.5,
            handlelength=2.0,
            columnspacing=2.0,
        )
        fig.add_artist(leg)

    if with_bs:
        markersize = {
            bs: x
            for (bs, x) in zip(expdef.ALL_BS, np.sqrt(np.linspace(1**2, 4.0**2, 5)))
        }
        lines = []
        for bs in reversed(expdef.ALL_BS):
            linestyle = {
                "linestyle": "",
                "marker": "v",
                "color": "grey",
                "markersize": markersize[bs],
            }
            lines.append(matplotlib.lines.Line2D([], [], **linestyle, label=bs))

        leg = fig.legend(
            handles=lines,
            loc="center right",
            ncol=len(expdef.ALL_BS),
            frameon=False,
            borderpad=1,
            handletextpad=-0.3,
            columnspacing=0.6,
        )
        fig.add_artist(leg)


if __name__ == "__main__":
    settings(plt)

    data = load_data()

    for normalized in [True, False]:
        for with_bs in [True, False]:
            fig = plt.figure()
            make_figure(fig, data, normalized=normalized, with_bs=with_bs)
            filename = Path(__file__).stem
            filename += "_normalized" if normalized else "_standard"
            filename += "_with_bs" if with_bs else ""
            plth.save(fig, name=os.path.join("output", filename))
            plt.close(fig)
