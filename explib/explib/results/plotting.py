import os
import pickle
import warnings
from datetime import datetime
from math import atan2, degrees
from pathlib import Path

import numpy
import numpy as np
import pandas as pd

# Label line with line2D label data
from explib import config
from explib.results import cleanup as datacleaning
from explib.results import data as data_h
from explib.results import experiment_groups as expdef
from matplotlib import ticker
from matplotlib.dates import date2num


def subsample_idx(length, n, log=False):
    """Returns a n-subset of [0,length-1]"""
    if log:
        log_grid = np.logspace(start=0, stop=np.log10(length - 1), num=n - 1)
        idx = [0] + list(log_grid.astype(int))
    else:
        lin_grid = np.linspace(start=0, stop=length - 1, num=n)
        idx = list(lin_grid.astype(int))
    idx = sorted(list(set(idx)))
    return idx


def subsample(xs, n=100, log=False):
    aslist = list(xs)
    return [aslist[i] for i in subsample_idx(len(aslist), n=n, log=False)]


def labelLine(
    line,
    x,
    label=None,
    align=True,
    drop_label=False,
    manual_rotation=0,
    ydiff=0.0,
    **kwargs,
):
    """Label a single matplotlib line at position x.

    Parameters
    ----------
    line : matplotlib.lines.Line
       The line holding the label
    x : number
       The location in data unit of the label
    label : string, optional
       The label to set. This is inferred from the line by default
    drop_label : bool, optional
       If True, the label is consumed by the function so that subsequent calls to e.g. legend
       do not use it anymore.
    kwargs : dict, optional
       Optional arguments passed to ax.text
    """
    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    mask = np.isfinite(ydata)
    if mask.sum() == 0:
        raise Exception("The line %s only contains nan!" % line)

    # Find first segment of xdata containing x
    if len(xdata) == 2:
        i = 0
        xa = min(xdata)
        xb = max(xdata)
    else:
        for i, (xa, xb) in enumerate(zip(xdata[:-1], xdata[1:])):
            if min(xa, xb) <= x <= max(xa, xb):
                break
        else:
            raise Exception("x label location is outside data range!")

    def x_to_float(x):
        """Make sure datetime values are properly converted to floats."""
        return date2num(x) if isinstance(x, datetime) else x

    xfa = x_to_float(xa)
    xfb = x_to_float(xb)
    ya = ydata[i]
    yb = ydata[i + 1]
    y = ya + (yb - ya) * (x_to_float(x) - xfa) / (xfb - xfa)

    if not (np.isfinite(ya) and np.isfinite(yb)):
        warnings.warn(
            (
                "%s could not be annotated due to `nans` values. "
                "Consider using another location via the `x` argument."
            )
            % line,
            UserWarning,
        )
        return

    if not label:
        label = line.get_label()

    if drop_label:
        line.set_label(None)

    if align:
        # Compute the slope and label rotation
        screen_dx, screen_dy = ax.transData.transform(
            (xfa, ya)
        ) - ax.transData.transform((xfb, yb))
        rotation = (degrees(atan2(screen_dy, screen_dx)) + 90) % 180 - 90
    else:
        rotation = manual_rotation

    # Set a bunch of keyword arguments
    if "color" not in kwargs:
        kwargs["color"] = line.get_color()

    if ("horizontalalignment" not in kwargs) and ("ha" not in kwargs):
        kwargs["ha"] = "center"

    if ("verticalalignment" not in kwargs) and ("va" not in kwargs):
        kwargs["va"] = "center"

    if "backgroundcolor" not in kwargs:
        kwargs["backgroundcolor"] = ax.get_facecolor()

    if "clip_on" not in kwargs:
        kwargs["clip_on"] = True

    if "zorder" not in kwargs:
        kwargs["zorder"] = 2.5

    ax.text(x, y + ydiff, label, rotation=rotation, **kwargs)


def rgb_to_unit(xs):
    """Convert a list of RGB numbers [1, 255] to a list of unit [0, 1]"""
    return [x / 255.0 for x in xs]


COLORS = {
    "Google Blue": {
        "color": "#4184f3",
        "active": "#3a53c5",
        "disabled": "#cad8fc",
    },
    "Google Red": {
        "color": "#db4437",
        "active": "#8f2a0c",
        "disabled": "#e8c6c1",
    },
    "Google Yellow": {
        "color": "#f4b400",
        "active": "#db9200",
        "disabled": "#f7e8b0",
    },
    "Google Green": {
        "color": "#0f9d58",
        "active": "#488046",
        "disabled": "#c2e1cc",
    },
    "Purple": {
        "color": "#aa46bb",
        "active": "#5c1398",
        "disabled": "#d7bce6",
    },
    "Teal": {
        "color": "#00abc0",
        "active": "#47828e",
        "disabled": "#c2eaf2",
    },
    "Deep Orange": {
        "color": "#ff6f42",
        "active": "#ca4a06",
        "disabled": "#f2cbba",
    },
    "Lime": {
        "color": "#9d9c23",
        "active": "#7f771d",
        "disabled": "#f1f4c2",
    },
    "Indigo": {
        "color": "#5b6abf",
        "active": "#3e47a9",
        "disabled": "#c5c8e8",
    },
    "Pink": {
        "color": "#ef6191",
        "active": "#ca1c60",
        "disabled": "#e9b9ce",
    },
    "Deep Teal": {
        "color": "#00786a",
        "active": "#2b4f43",
        "disabled": "#bededa",
    },
    "Deep Pink": {
        "color": "#c1175a",
        "active": "#75084f",
        "disabled": "#de8cae",
    },
    "Gray": {
        "color": "#9E9E9E",
        "active": "#424242",
        "disabled": "F5F5F5",
    },
    "VB": {
        "blue": rgb_to_unit([0, 119, 187]),
        "red": rgb_to_unit([204, 51, 17]),
        "orange": rgb_to_unit([238, 119, 51]),
        "cyan": rgb_to_unit([51, 187, 238]),
        "teal": rgb_to_unit([0, 153, 136]),
        "magenta": rgb_to_unit([238, 51, 119]),
        "grey": rgb_to_unit([187, 187, 187]),
    },
    "PTyellow": rgb_to_unit([221, 170, 51]),
    "PTred": rgb_to_unit([187, 85, 102]),
    "PTblue": rgb_to_unit([0, 68, 136]),
    "PTMC": {
        "lightyellow": "#EECC66",
        "lightred": "#EE99AA",
        "lightblue": "#6699CC",
        "darkyellow": "#997700",
        "darkred": "#994455",
        "darkblue": "#004488",
    },
}


GOOGLE_STYLE_COLORS = {
    "b1": COLORS["Google Blue"]["color"],
    "b2": COLORS["Google Blue"]["active"],
    "g1": COLORS["Google Green"]["color"],
    "g2": COLORS["Google Green"]["active"],
    "t1": COLORS["Google Yellow"]["color"],
    "t2": COLORS["Google Yellow"]["active"],
    "gr1": COLORS["Gray"]["color"],
    "gr2": COLORS["Gray"]["active"],
    "p1": COLORS["Deep Pink"]["color"],
    "p2": COLORS["Deep Pink"]["active"],
    "r1": COLORS["VB"]["orange"],
    "r2": COLORS["VB"]["red"],
    "gray": "#808080",
    "black": "#000000",
}

BASE_COLORS = {
    "b1": [0, 0, 0],  # COLORS["PTblue"],
    "b2": [0, 0, 0],  # COLORS["PTblue"],
    "r1": COLORS["PTred"],
    "r2": COLORS["PTred"],
    "t1": COLORS["PTyellow"],
    "t2": COLORS["PTyellow"],
    "gr1": COLORS["Gray"]["color"],
    "gr2": COLORS["Gray"]["active"],
    "g1": COLORS["PTblue"],
    "g2": COLORS["PTblue"],
    "p1": COLORS["Deep Pink"]["color"],
    "p2": COLORS["Deep Pink"]["active"],
    "gray": "#808080",
    "black": "#000000",
}


# Magic constants
_stroke_width = 0.5
_xtick_width = 0.8
_GOLDEN_RATIO = (5.0**0.5 - 1.0) / 2.0


def base_font(*, family="sans-serif"):
    # ptmx replacement
    fontset = "stix" if family == "serif" else "stixsans"
    return {
        "text.usetex": False,
        "font.sans-serif": ["TeX Gyre Heros"],
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": fontset,
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "font.family": family,
    }


fontsizes = {
    "normal": 9,
    "small": 7,
    "tiny": 6,
}


def base_fontsize(*, base=10):
    fontsizes = {
        "normal": base - 1,
        "small": base - 3,
        "tiny": base - 4,
    }

    return {
        "font.size": fontsizes["normal"],
        "axes.titlesize": fontsizes["normal"],
        "axes.labelsize": fontsizes["small"],
        "legend.fontsize": fontsizes["small"],
        "xtick.labelsize": fontsizes["tiny"],
        "ytick.labelsize": fontsizes["tiny"],
    }


def base_layout(
    *,
    rel_width=1.0,
    nrows=1,
    ncols=2,
    constrained_layout=False,
    tight_layout=False,
    height_to_width_ratio=_GOLDEN_RATIO,
    base_width_in=5.5,
):
    width_in = base_width_in * rel_width
    subplot_width_in = width_in / ncols
    subplot_height_in = height_to_width_ratio * subplot_width_in
    height_in = subplot_height_in * nrows
    figsize = (width_in, height_in)

    return {
        "figure.dpi": 250,
        "figure.figsize": figsize,
        "figure.constrained_layout.use": constrained_layout,
        "figure.autolayout": tight_layout,
        # Padding around axes objects. Float representing
        "figure.constrained_layout.h_pad": 1 / 72,
        # inches. Default is 3/72 inches (3 points)
        "figure.constrained_layout.w_pad": 1 / 72,
        # Space between subplot groups. Float representing
        "figure.constrained_layout.hspace": 0.00,
        # a fraction of the subplot widths being separated.
        "figure.constrained_layout.wspace": 0.00,
    }


def base_style():
    grid_color = BASE_COLORS["gray"]
    text_color = BASE_COLORS["black"]
    return {
        "text.color": text_color,
        "axes.labelcolor": text_color,
        "axes.labelpad": 2,
        #        "axes.spines.left": False,
        #        "axes.spines.bottom": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        #        # "ytick.minor.left": False,
        #        # Axes aren't used in this theme, but still set some properties in case the user
        #        # decides to turn them on.
        "axes.edgecolor": grid_color,
        "axes.linewidth": _stroke_width,
        # default is "line", i.e., below lines but above patches (bars)
        #        "axes.axisbelow": True,
        #        #
        #        "ytick.right": False,
        #        "ytick.color": grid_color,
        #        "ytick.labelcolor": text_color,
        #        "ytick.major.width": _stroke_width,
        #        "xtick.minor.top": False,
        #        "xtick.minor.bottom": False,
        #        "xtick.color": grid_color,
        #        "xtick.labelcolor": text_color,
        #        "xtick.major.width": _xtick_width,
        #        "axes.grid": True,
        #        "axes.grid.axis": "y",
        "ytick.major.pad": 1,
        "xtick.major.pad": 1,
        "grid.color": grid_color,
        # Choose the line width such that it's very subtle, but still serves as a guide.
        "grid.linewidth": _stroke_width,
        "axes.xmargin": 0,
        "axes.ymargin": 0,
        "axes.titlepad": 3,
    }


def smaller_style():
    return {
        "axes.labelpad": 2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "ytick.major.pad": 1,
        "xtick.major.pad": 1,
        "axes.xmargin": 0,
        "axes.ymargin": 0,
        "axes.titlepad": 3,
    }


def base_config(
    *,
    rel_width=1.0,
    nrows=1,
    ncols=4,
    family="serif",
    height_to_width_ratio=_GOLDEN_RATIO,
):
    font_config = base_font(family=family)
    fonsize_config = base_fontsize(base=10)
    layout_config = base_layout(
        rel_width=rel_width,
        nrows=nrows,
        ncols=ncols,
        height_to_width_ratio=height_to_width_ratio,
    )
    style_config = base_style()
    return {**font_config, **fonsize_config, **layout_config, **style_config}


def smaller_config(
    *,
    rel_width=1.0,
    nrows=1,
    ncols=4,
    family="sans-serif",
    height_to_width_ratio=_GOLDEN_RATIO,
):
    font_config = base_font(family=family)
    fonsize_config = base_fontsize(base=10)
    layout_config = base_layout(
        rel_width=rel_width,
        nrows=nrows,
        ncols=ncols,
        height_to_width_ratio=height_to_width_ratio,
    )
    style_config = smaller_style()
    return {**font_config, **fonsize_config, **layout_config, **style_config}


def iclr_config_2(
    *,
    rel_width=1.0,
    nrows=1,
    ncols=4,
    family="sans-serif",
    height_to_width_ratio=_GOLDEN_RATIO,
):
    font_config = base_font(family=family)
    fonsize_config = base_fontsize(base=11)
    layout_config = base_layout(
        rel_width=rel_width,
        nrows=nrows,
        ncols=ncols,
        height_to_width_ratio=height_to_width_ratio,
        base_width_in=5.5,
    )
    style_config = smaller_style()
    return {**font_config, **fonsize_config, **layout_config, **style_config}


def icml_config(
    *,
    rel_width=1.0,
    nrows=1,
    ncols=4,
    family="sans-serif",
    height_to_width_ratio=_GOLDEN_RATIO,
):
    font_config = base_font(family=family)
    fonsize_config = base_fontsize(base=11)
    layout_config = base_layout(
        rel_width=rel_width,
        nrows=nrows,
        ncols=ncols,
        height_to_width_ratio=height_to_width_ratio,
        base_width_in=6.75,
    )
    style_config = smaller_style()
    return {**font_config, **fonsize_config, **layout_config, **style_config}


def save(fig, name, tight=False, transparent=False):
    print(f"Saving figure {name}.pdf")
    fig.savefig(
        f"{name}.pdf",
        bbox_inches="tight" if tight else None,
        transparent=transparent,
    )


def hide_frame(*axes, top=True, right=True, left=False, bottom=False):
    for ax in axes:
        ax.spines["top"].set_visible(not top)
        ax.spines["right"].set_visible(not right)
        ax.spines["left"].set_visible(not left)
        ax.spines["bottom"].set_visible(not bottom)


def hide_all_frame(*axes):
    hide_frame(*axes, top=True, right=True, left=True, bottom=True)


def hide_ticklabels(*axes, x=True, y=True):
    for ax in axes:
        if x:
            ax.set_xticklabels([], minor=True)
            ax.set_xticklabels([], minor=False)
        if y:
            ax.set_yticklabels([], minor=True)
            ax.set_yticklabels([], minor=False)


def hide_ticks(*axes, x=True, y=True):
    for ax in axes:
        if x:
            ax.set_xticks([], minor=True)
            ax.set_xticks([], minor=False)
        if y:
            ax.set_yticks([], minor=True)
            ax.set_yticks([], minor=False)


def flatten(t):
    return [item for sublist in t for item in sublist]


##
#

displaynames = {
    "small": "S",
    "medium": "M",
    "large": "L",
    "larger": "XL",
    "full": "Full",
    "resnet18": "ResNet18",
    "transformer_encoder": "Transformer",
    "transformer_xl": "Transformer XL",
    "distilbert_base_pretrained": "DistilBert",
    "mnist": "MNIST",
    "cifar10": "CIFAR-10",
    "ptb": "PTB",
    "wikitext2": "WikiText-2",
    "squad": "SQuAD",
    "sgd": "SGD",
    "adam": "Adam",
    "sgd+m": "SGD($+$m)",
    "adam+m": "Adam($+$m)",
    "sgd-m": "SGD($-$m)",
    "adam-m": "Adam($-$m)",
    "training_loss": "Train Loss",
    "training_perf": "Train Perf.",
    "validation_perf": "Val. Perf.",
    "PlainSGD": "GD",
    "SGD": "SGD",
    "NormalizedGD": "Norm. GD($-$m)",
    "NormalizedGD+m": "Norm. GD($+$m)",
    "BlockNormalizedGD": "Block-Normalized",
    "SignDescent": "Sign descent($-$m)",
    "SignDescent+m": "Sign descent($+$m)",
    "RescaledSignDescent": "Rescaled SD($-$m)",
    "RescaledSignDescent+m": "Rescaled SD($+$m)",
    "train_accuracy": "Train Acc.",
    "train_ppl": "Train PPL",
    "train_exact_f1": "Train F1",
    "valid_accuracy": "Valid Acc.",
    "valid_ppl": "Valid PPL",
    "valid_exact_f1": "Valid F1",
}

abbrevs_ = {
    "sgd": "SGD",
    "adam": "Adam",
    "sgd+m": "SGD($+$m)",
    "adam+m": "Adam($+$m)",
    "sgd-m": "SGD($-$m)",
    "adam-m": "Adam($-$m)",
    "NormalizedGD": "Norm.($-$m)",
    "NormalizedGD+m": "Norm.($+$m)",
    "BlockNormalizedGD": "Block",
    "SignDescent": "Sign($-$m)",
    "SignDescent+m": "Sign($+$m)",
    "accuracy": "Acc.",
    "ppl": "PPL",
    "f1": "F1",
    "train_accuracy": "Acc.",
    "train_ppl": "PPL",
    "train_exact_f1": "F1",
    "valid_accuracy": "Acc.",
    "valid_ppl": "PPL",
    "valid_exact_f1": "F1",
}


def abbrev(key):
    return abbrevs_.get(key, key)


def fdisplaynames(key):
    if key in displaynames:
        return displaynames[key]
    else:
        return key


markersize_small = 3
linewidth_small = 1

linestyles = {
    "sgd+m": {
        "marker": ".",
        "markersize": markersize_small,
        "linewidth": linewidth_small,
        "linestyle": "-",
        "color": BASE_COLORS["b2"],
    },
    "sgd-m": {
        "marker": ".",
        "markersize": markersize_small,
        "linewidth": linewidth_small,
        "linestyle": "--",
        "dashes": (4, 5),
        "color": BASE_COLORS["b1"],
    },
    "adam+m": {
        "marker": ".",
        "markersize": markersize_small,
        "linewidth": linewidth_small,
        "linestyle": "-",
        "color": BASE_COLORS["r2"],
    },
    "adam-m": {
        "marker": ".",
        "markersize": markersize_small,
        "linewidth": linewidth_small,
        "linestyle": "--",
        "dashes": (4, 5),
        "color": BASE_COLORS["r1"],
    },
    "NormalizedGD": {
        "marker": ".",
        "markersize": markersize_small,
        "linewidth": linewidth_small,
        "linestyle": "--",
        "dashes": (4, 5),
        "color": BASE_COLORS["t1"],
    },
    "BlockNormalizedGD": {
        "marker": ".",
        "markersize": markersize_small,
        "linewidth": linewidth_small,
        "linestyle": "--",
        "color": BASE_COLORS["t2"],
    },
    "SignDescent": {
        "marker": ".",
        "markersize": markersize_small,
        "linewidth": linewidth_small,
        "linestyle": "--",
        "dashes": (4, 5),
        "color": BASE_COLORS["g2"],
    },
    "RescaledSignDescent": {
        "marker": ".",
        "linewidth": linewidth_small,
        "markersize": markersize_small,
        "linestyle": "--",
        "color": BASE_COLORS["g1"],
    },
    "NormalizedGD+m": {
        "marker": ".",
        "markersize": markersize_small,
        "linewidth": linewidth_small,
        "linestyle": "-",
        "color": BASE_COLORS["t1"],
    },
    "BlockNormalizedGD+m": {
        "marker": ".",
        "markersize": markersize_small,
        "linewidth": linewidth_small,
        "linestyle": "-",
        "color": BASE_COLORS["t2"],
    },
    "SignDescent+m": {
        "marker": ".",
        "markersize": markersize_small,
        "linewidth": linewidth_small,
        "linestyle": "-",
        "color": BASE_COLORS["g2"],
    },
    "RescaledSignDescent+m": {
        "marker": ".",
        "linewidth": linewidth_small,
        "markersize": markersize_small,
        "linestyle": "-",
        "color": BASE_COLORS["g1"],
    },
}

linestyles = {
    **linestyles,
    "adam+m+bc": linestyles["sgd+m"],
    "adam-m+bc": linestyles["sgd-m"],
    "adam+m-bc": linestyles["adam+m"],
    "adam-m-bc": linestyles["adam-m"],
    "SGD": linestyles["sgd-m"],
}
for b2 in [".9", ".6", ".3", ".1", ".0"]:
    linestyles[f"adam+m+bc_b2={b2}"] = linestyles["adam+m+bc"]
    linestyles[f"adam-m+bc_b2={b2}"] = linestyles["adam-m+bc"]
    linestyles[f"adam+m-bc_b2={b2}"] = linestyles["adam+m-bc"]
    linestyles[f"adam-m-bc_b2={b2}"] = linestyles["adam-m-bc"]

linestyles_nm = {}
for key, style in linestyles.items():
    style_nm = {**style}
    style_nm["marker"] = None
    linestyles_nm[key] = style_nm

fillstyles = {
    k: {"color": linestyles[k]["color"], "alpha": 0.2} for k in linestyles.keys()
}

fillstyles = {
    **fillstyles,
    "adam+m+bc": fillstyles["sgd+m"],
    "adam-m+bc": fillstyles["sgd-m"],
    "adam+m-bc": fillstyles["adam+m"],
    "adam-m-bc": fillstyles["adam-m"],
    "SGD": fillstyles["sgd-m"],
    # "NormalizedGD": fillstyles["sgd+m"],
    # "BlockNormalizedGD": fillstyles["sgd-m"],
    # "SignDescent": fillstyles["adam+m"],
    # "RescaledSignDescent": fillstyles["adam-m"],
}
for b2 in [".9", ".6", ".3", ".1", ".0"]:
    fillstyles[f"adam+m+bc_b2={b2}"] = fillstyles["adam+m+bc"]
    fillstyles[f"adam-m+bc_b2={b2}"] = fillstyles["adam-m+bc"]
    fillstyles[f"adam+m-bc_b2={b2}"] = fillstyles["adam+m-bc"]
    fillstyles[f"adam-m-bc_b2={b2}"] = fillstyles["adam-m-bc"]

opt_names = ["sgd-m", "sgd+m", "adam-m", "adam+m"]
normalized_opt_names = [
    "SGD",
    "NormalizedGD",
    "BlockNormalizedGD",
    "SignDescent",
    "RescaledSignDescent",
]
opt_names_normonly = [
    "NormalizedGD",
    "SignDescent",
    "RescaledSignDescent",
    "NormalizedGD+m",
    "SignDescent+m",
    "RescaledSignDescent+m",
]
opt_filters = {
    "sgd-m": {"opt.name": "SGD", "opt.momentum": 0.0},
    "sgd+m": {"opt.name": "SGD", "opt.momentum": 0.9},
    "adam+m": {"opt.name": "Adam", "opt.b1": 0.9},
    "adam-m": {"opt.name": "Adam", "opt.b1": 0.0},
    "SGD": {"opt.name": "SGD"},
    "SignDescent": {"opt.name": "SignDescent", "opt.momentum": 0.0},
    "NormalizedGD": {"opt.name": "NormalizedGD", "opt.momentum": 0.0},
    "BlockNormalizedGD": {"opt.name": "BlockNormalizedGD", "opt.momentum": 0.0},
    "RescaledSignDescent": {
        "opt.name": "RescaledSignDescent",
        "opt.momentum": 0.0,
        "opt.norm": 1.0,
    },
    "SignDescent+m": {"opt.name": "SignDescent", "opt.momentum": 0.9},
    "NormalizedGD+m": {"opt.name": "NormalizedGD", "opt.momentum": 0.9},
    "BlockNormalizedGD+m": {"opt.name": "BlockNormalizedGD", "opt.momentum": 0.9},
    "RescaledSignDescent+m": {
        "opt.name": "RescaledSignDescent",
        "opt.momentum": 0.9,
        "opt.norm": 1.0,
    },
}

problems = {
    "LEN": {"model": "lenet5", "dataset": "mnist"},
    "RES": {"model": "resnet18", "dataset": "cifar10"},
    "TEC": {"model": "transformer_encoder", "dataset": "ptb"},
    "TXL": {"model": "transformer_xl", "dataset": "wikitext2"},
    "BRT": {"model": "distilbert_base_pretrained", "dataset": "squad"},
}

models_datasets = [
    ("lenet5", "mnist"),
    ("resnet18", "cifar10"),
    ("transformer_encoder", "ptb"),
    ("distilbert_base_pretrained", "squad"),
]


def make_textbf(str):
    return r"$\mathrm{\bf" + str.replace(" ", r"\,") + r"}}$"


def compute_limits(data_min, data_max, margin_p=0.05, logspace=True):
    def get_limits(data_min, data_max, margin_p=0.05):
        dp = margin_p
        """
        Given data min and max, returns limits lim- and lim+
        such that [min,max] is centered in [lim-, lim+]
        and represent (1-2*margin_p) percent of the interval such that

        lim- + (1-margin_p) * (lim+ - lim-) = data_max
        lim- + margin_p * (lim+ - lim-) = data_min
        """
        W = np.array([[1 - dp, dp], [dp, 1 - dp]]).reshape((2, 2))
        data_values = np.array([data_min, data_max]).reshape((-1, 1))
        limits = np.linalg.solve(W, data_values)

        assert np.allclose(limits[0] + (limits[1] - limits[0]) * (1 - dp), data_max)
        assert np.allclose(limits[0] + (limits[1] - limits[0]) * dp, data_min)

        return limits[0][0], limits[1][0]

    if not logspace:
        return get_limits(data_min, data_max, margin_p)
    else:
        x0, x1 = np.log(data_min), np.log(data_max)
        z0, z1 = get_limits(x0, x1, margin_p)
        return np.exp(z0), np.exp(z1)


def find_best_stepsize(rundata, at_epoch=None, by_key="training_loss_runs"):
    if at_epoch is not None:
        rundata = data_h.df_select(rundata, epoch_runs=at_epoch)

    medians, mins, maxs, xs = data_h.median_min_max_by(
        rundata, key="opt.alpha", metric_name=by_key
    )

    best_alpha_idx = np.nanargmin(maxs)
    best_alpha = xs[best_alpha_idx]
    return best_alpha


def plot_metric_by_key(ax, metric_name, optname, rundata, key="opt.alpha"):
    medians, mins, maxs, xs = data_h.median_min_max_by(
        rundata, key=key, metric_name=metric_name
    )

    ax.plot(
        xs,
        medians,
        **linestyles[optname],
        label=displaynames[optname],
    )
    ax.fill_between(
        xs,
        mins,
        maxs,
        **fillstyles[optname],
    )


def plot_optim_by_stepsize(ax, metric_name, optname, rundata):
    medians, mins, maxs, xs = data_h.median_min_max_by(
        rundata, key="opt.alpha", metric_name=metric_name
    )

    ax.plot(
        xs,
        medians,
        **linestyles[optname],
        label=fdisplaynames(optname),
    )
    ax.fill_between(
        xs,
        mins,
        maxs,
        **fillstyles[optname],
    )

    best_alpha = find_best_stepsize(rundata)
    best_alpha_idx = np.where(xs == best_alpha)[0][0]
    ax.plot(
        xs[best_alpha_idx],
        medians[best_alpha_idx],
        color="k",
        marker="*",
        markersize=3,
        zorder=5,
    )


def higher_is_better(metric_name):
    higher = ["accuracy", "f1"]
    lower = ["ppl", "loss"]
    if any([x in metric_name for x in higher]):
        return True
    if any([x in metric_name for x in lower]):
        return False
    raise NotImplementedError(
        f"Metric {metric_name} unknown, can't determine if higher is better."
        + f"Only know of higher: {higher}, lower: {lower}"
    )


def should_log(metric_name):
    to_log = ["ppl", "loss"]
    not_log = ["accuracy", "f1"]
    if any([x in metric_name for x in to_log]):
        return True
    if any([x in metric_name for x in not_log]):
        return False
    raise NotImplementedError(
        f"Metric {metric_name} unknown, can't determine if log scale."
        + f"Only know of to_log: {to_log}, lower: {not_log}"
    )


def get_metric_at_start_for_dataset(runs, summaries, dataset, metric):
    runs_at_start = datacleaning.filter_merge(
        summaries, runs, summary_filter={"dataset": dataset}, runs_filter={"epoch": 0}
    )
    return runs_at_start[f"{metric}_runs"].mean()


def get_metric_limits_for_dataset(runs, summaries, dataset, metric, margin_p=0.2):
    if "accuracy" in metric or "f1" in metric:
        return [0, 105.0]

    all_runs = datacleaning.filter_merge(
        summaries, runs, summary_filter={"dataset": dataset}, runs_filter={}
    )
    start_val = get_metric_at_start_for_dataset(runs, summaries, dataset, metric)
    # Replace values of training loss == 0 by nan so we get the second-smallest elem
    all_runs["training_loss_runs"] = all_runs["training_loss_runs"].where(
        all_runs["training_loss_runs"] != 0.0, np.nan
    )
    min_val = all_runs[f"{metric}_runs"].min()
    max_val = all_runs[f"{metric}_runs"].max()

    try:
        if higher_is_better(metric):
            return compute_limits(start_val, max_val, margin_p=margin_p)
        else:
            return compute_limits(min_val, start_val, margin_p=margin_p)
    except AssertionError:
        import pdb

        pdb.set_trace()


def accumulate_fitler(accumulate):
    return {} if accumulate is None else {"accumulate_steps": accumulate}


def eff_bs(bs, accum):
    return bs * (accum if accum is not None else 1)


def clip_metric_at(df_, key, ylims):
    scaling = 2 if should_log(key) else 1.05

    if higher_is_better(key):
        val_ = (1 / scaling) * ylims[0]
        df_[key] = np.where(df_[key].isna(), val_, df_[key])
        df_[key] = np.where(df_[key] < val_, val_, df_[key])
    else:
        val_ = scaling * ylims[1]
        df_[key] = np.where(df_[key].isna(), val_, df_[key])
        df_[key] = np.where(df_[key] > val_, val_, df_[key])

    if "accuracy" in key or "f1" in key:
        df_[key] = np.where(df_[key] < 0, 0, df_[key])
        df_[key] = np.where(df_[key] > 100, 100, df_[key])

    return df_


metric_type_to_dset_to_metric = {
    "training_loss": {
        "mnist": "training_loss",
        "cifar10": "training_loss",
        "ptb": "training_loss",
        "wikitext2": "training_loss",
        "squad": "training_loss",
    },
    "training_perf": {
        "mnist": "train_accuracy",
        "cifar10": "train_accuracy",
        "ptb": "train_ppl",
        "wikitext2": "train_ppl",
        "squad": "train_exact_f1",
    },
    "validation_perf": {
        "mnist": "valid_accuracy",
        "cifar10": "valid_accuracy",
        "ptb": "valid_ppl",
        "wikitext2": "valid_ppl",
        "squad": "valid_exact_f1",
    },
}


experiment_settings = {
    "fix-full-batch-training-squad": {
        "squad": {
            "full": {
                "clip_epoch": 60,
                "max_epoch": 80,
                "batch_size": 16,
                "accumulate_steps": 1370 * 4,
            },
        },
    },
    "no-dropout": {
        "wikitext2": {
            "full": {
                "clip_epoch": 320,
                "max_epoch": 320,
                "batch_size": 80,
                "accumulate_steps": 203,
            },
        },
        "ptb": {
            "full": {
                "clip_epoch": 800 * 4,
                "max_epoch": 800 * 4,
                "batch_size": 1326,
                "accumulate_steps": 20,
            },
        },
    },
    "full_batch": {
        "clip_epoch": {
            "mnist": 800,
            "cifar10": 800,
            "ptb": 3200,
            "wikitext2": 320,
            "squad": 60,
        }
    },
    "norm-ablation": {
        "mnist": {
            "medium": {"clip_epoch": 100, "max_epoch": 100, "batch_size": 1024},
            "large": {"clip_epoch": 200, "max_epoch": 200, "batch_size": 4096},
            "larger": {"clip_epoch": 400, "max_epoch": 800, "batch_size": 16384},
        },
        "cifar10": {
            "medium": {"clip_epoch": 100, "max_epoch": 100, "batch_size": 256},
            "large": {"clip_epoch": 100, "max_epoch": 200, "batch_size": 1024},
            "larger": {"clip_epoch": 200, "max_epoch": 400, "batch_size": 4096},
        },
        "ptb": {
            "medium": {"clip_epoch": 100, "max_epoch": 100, "batch_size": 64},
            "large": {"clip_epoch": 200, "max_epoch": 200, "batch_size": 256},
            "larger": {"clip_epoch": 400, "max_epoch": 800, "batch_size": 1024},
        },
        "wikitext2": {
            "medium": {
                "clip_epoch": 40,
                "max_epoch": 40,
                "batch_size": 80,
                "accumulate_steps": 1,
            },
            "large": {
                "clip_epoch": 40,
                "max_epoch": 80,
                "batch_size": 80,
                "accumulate_steps": 4,
            },
            "larger": {
                "clip_epoch": 80,
                "max_epoch": 160,
                "batch_size": 80,
                "accumulate_steps": 16,
            },
        },
        "squad": {
            "medium": {
                "clip_epoch": 5,
                "max_epoch": 10,
                "batch_size": 16,
                "accumulate_steps": 2,
            },
            "large": {
                "clip_epoch": 5,
                "max_epoch": 10,
                "batch_size": 16,
                "accumulate_steps": 32,
            },
            "larger": {
                "clip_epoch": 5,
                "max_epoch": 10,
                "batch_size": 16,
                "accumulate_steps": 128,
            },
        },
    },
    "norm-ablation-full": {
        "clip_epoch": {
            "mnist": 800,
            "cifar10": 800,
            "ptb": 3200,
            "wikitext2": 320,
            "squad": 60,
        },
        "max_epoch": {
            "mnist": 800,
            "cifar10": 800,
            "ptb": 3200,
            "wikitext2": 320,
            "squad": 80,
        },
    },
    "increasing_batch_size": {
        "problem_filters": {
            "mnist": [
                {"batch_size": 256, "max_epoch": 100},
                {"batch_size": 1024, "max_epoch": 100},
                {"batch_size": 4096, "max_epoch": 200},
                {"batch_size": 16384, "max_epoch": 800},
            ],
            "cifar10": [
                {"batch_size": 64, "max_epoch": 100},
                {"batch_size": 256, "max_epoch": 100},
                {"batch_size": 1024, "max_epoch": 200},
                {"batch_size": 4096, "max_epoch": 800},
            ],
            "ptb": [
                {"batch_size": 16, "max_epoch": 100},
                {"batch_size": 64, "max_epoch": 100},
                {"batch_size": 256, "max_epoch": 200},
                {"batch_size": 1024, "max_epoch": 800},
            ],
            "wikitext2": [
                {"batch_size": 20, "max_epoch": 40, "accumulate_steps": 1},
                {"batch_size": 80, "max_epoch": 40, "accumulate_steps": 1},
                {"batch_size": 80, "max_epoch": 80, "accumulate_steps": 4},
                {"batch_size": 80, "max_epoch": 160, "accumulate_steps": 16},
            ],
            "squad": [
                {"batch_size": 32, "max_epoch": 5, "accumulate_steps": 1},
                {"batch_size": 32, "max_epoch": 5, "accumulate_steps": 4},
                {"batch_size": 32, "max_epoch": 5, "accumulate_steps": 16},
                {"batch_size": 32, "max_epoch": 20, "accumulate_steps": 64},
            ],
        },
        "run_filters": {
            "mnist": [
                {"epoch": 100},
                {"epoch": 100},
                {"epoch": 200},
                {"epoch": 400},
            ],
            "cifar10": [
                {"epoch": 100},
                {"epoch": 100},
                {"epoch": 100},
                {"epoch": 200},
            ],
            "ptb": [
                {"epoch": 100},
                {"epoch": 100},
                {"epoch": 200},
                {"epoch": 400},
            ],
            "wikitext2": [
                {"epoch": 40},
                {"epoch": 40},
                {"epoch": 40},
                {"epoch": 80},
            ],
            "squad": [
                {"epoch": 5},
                {"epoch": 5},
                {"epoch": 5},
                {"epoch": 5},
            ],
        },
    },
}


def normalize_y_axis(*axes):
    miny, maxy = np.inf, -np.inf
    for ax in axes:
        y1, y2 = ax.get_ylim()
        miny = np.min([miny, y1])
        maxy = np.max([maxy, y2])
    for ax in axes:
        ax.set_ylim([miny, maxy])


def same_xlims(*axes):
    minx, maxx = np.inf, -np.inf
    for ax in axes:
        y1, y2 = ax.get_xlim()
        minx = np.min([minx, y1])
        maxx = np.max([maxx, y2])
    for ax in axes:
        ax.set_xlim([minx, maxx])


def make_yaxis_scale_and_ticks(ax, metric_type, dataset, data_ylim, special=None):
    metric = metric_type_to_dset_to_metric[metric_type][dataset]

    if should_log(metric):
        ax.set_yscale("log", base=10)
        ax.set_ylim(data_ylim)
        ax.yaxis.set_major_locator(ticker.LogLocator(numticks=4))
        ax.yaxis.set_minor_locator(ticker.LogLocator(numticks=4))

        if dataset == "ptb":
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.yaxis.set_ticks([4, 8], major=True)
            ax.yaxis.set_ticks([3, 4, 5, 6, 7, 8, 9], minor=True)
        if dataset == "squad":
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.yaxis.set_ticks([2, 4], major=True)
            ax.yaxis.set_ticks([2, 3, 4, 5, 6], minor=True)

    if special == "full batch":
        if dataset == "cifar10":
            ax.set_ylim([10**-5, 10**2])


# %%
# Dataloader shortcuts


def load_gradnorms(seed=0):
    files = {
        "mnist_1": f"norm_lenet5_mnist_1_{seed}_Adam_0.npy",
        "mnist_256": f"norm_lenet5_mnist_256_{seed}_Adam_0.npy",
        "cifar10_1": f"norm_resnet18_cifar10_2_{seed}_Adam_0.npy",
        "cifar10_64": f"norm_resnet18_cifar10_64_{seed}_Adam_0.npy",
        "ptb_1": f"norm_transformer_encoder_ptb_1_{seed}_Adam_0.npy",
        "ptb_16": f"norm_transformer_encoder_ptb_16_{seed}_Adam_0.npy",
        "wt2_1": f"norm_transformer_xl_wikitext2_1_{seed}_Adam_0.npy",
        "wt2_16": f"norm_transformer_xl_wikitext2_16_{seed}_Adam_0.npy",
        "squad_1": f"norm_distilbert_base_pretrained_squad_1_{seed}_SGD_0.npy",
        "squad_16": f"norm_distilbert_base_pretrained_squad_16_{seed}_SGD_0.npy",
    }
    data_dir = os.path.join(config.get_workspace(), "norms_and_text_data", "norms")

    return {
        dset: np.load(os.path.join(data_dir, dset, file))
        for dset, file in files.items()
    }


def preprocessed_file(filename):
    preprocessed_di = os.path.join(
        config.get_workspace(), "preprocessed_results_for_plotting"
    )
    Path(preprocessed_di).mkdir(parents=True, exist_ok=True)
    return os.path.join(preprocessed_di, filename)


def save_preprocessed(data, filename):
    with open(preprocessed_file(filename), "wb") as fh:
        pickle.dump(data, fh)


def load_preprocessed(filename):
    with open(preprocessed_file(filename), "rb") as fh:
        return pickle.load(fh)


markers_loglog = ["o", "s"]


YLIM_TRAINLOSS_LOG = {
    expdef.MNIST: [10**-6, 10**1],
    expdef.CIFAR10: [10**-7, 10**2],
    expdef.PTB: [1.7, 7],
    expdef.WT2: [10**-1, 10**1.1],
    expdef.SQUAD: [10**-1, 10**1.1],
}
YLIM_TRAINPERF_LOG = {
    expdef.MNIST: [0, 110],
    expdef.CIFAR10: [0, 110],
    expdef.PTB: [10**0, 10**4.5],
    expdef.WT2: [10**-0.5, 10**5],
    expdef.SQUAD: [0, 110],
}

INIT_LOSSES = {
    expdef.MNIST: 2.306351,
    expdef.CIFAR10: 6.976949,
    expdef.PTB: 9.270155,
    expdef.SQUAD: 5.533700,
    expdef.WT2: 11.182902,
}
MIN_LOSSES = {
    expdef.MNIST: 6.762069e-07,
    expdef.CIFAR10: 8.835905e-09,
    expdef.PTB: 1.922758e00,
    expdef.SQUAD: 8.793967e-03,
    expdef.WT2: 4.584191e-02,
}


def get_min_max(ax, axis="x"):
    """Returns the min and max values of the "x" or "y" axis for lines in
    ax."""
    vals = [v for line in ax.lines for v in (line._x if axis == "x" else line._y)]
    return np.min(vals), np.max(vals)


def make_grid(fig, grid_type="2-3"):
    from matplotlib import gridspec

    if grid_type == "2x2-3":

        gs_base = fig.add_gridspec(
            nrows=1,
            ncols=2,
            width_ratios=(2, 3 + 1 / 8),
            left=0.06,
            right=0.99,
            bottom=0.085,
            top=0.925,
            wspace=0.175,
            hspace=0.5,
        )
        gs_left = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=gs_base[0], wspace=0.375, hspace=0.425
        )
        gs_right = gridspec.GridSpecFromSubplotSpec(
            2, 3, subplot_spec=gs_base[1], wspace=0.375, hspace=0.425
        )
        axes = [
            [
                fig.add_subplot(gs_left[0, 0]),
                fig.add_subplot(gs_left[0, 1]),
                fig.add_subplot(gs_right[0, 0]),
                fig.add_subplot(gs_right[0, 1]),
                fig.add_subplot(gs_right[0, 2]),
            ],
            [
                fig.add_subplot(gs_left[1, 0]),
                fig.add_subplot(gs_left[1, 1]),
                fig.add_subplot(gs_right[1, 0]),
                fig.add_subplot(gs_right[1, 1]),
                fig.add_subplot(gs_right[1, 2]),
            ],
        ]
    if grid_type == "2-3":
        gs_base = fig.add_gridspec(
            nrows=1,
            ncols=2,
            width_ratios=(2, 3 + 1 / 8 + 1 / 32),
            left=0.0775,
            right=0.99,
            bottom=0.19,
            top=0.84,
            wspace=0.15,
            hspace=0.10,
        )
        gs_left = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs_base[0], wspace=0.375, hspace=0.7
        )
        gs_right = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=gs_base[1], wspace=0.375, hspace=0.7
        )
        axes = [
            [
                fig.add_subplot(gs_left[0, 0]),
                fig.add_subplot(gs_left[0, 1]),
                fig.add_subplot(gs_right[0, 0]),
                fig.add_subplot(gs_right[0, 1]),
                fig.add_subplot(gs_right[0, 2]),
            ]
        ]
    return axes


def make_grid_iclr(fig, grid_type="2-3", tight=False):
    from matplotlib import gridspec

    axes = None
    if grid_type == "2x2":
        gs_base = fig.add_gridspec(
            nrows=2,
            ncols=2,
            left=0.08,
            right=0.98,
            bottom=0.12,
            top=0.94,
            wspace=0.2,
            hspace=0.3,
        )
        axes = [
            [
                fig.add_subplot(gs_base[0, 0]),
                fig.add_subplot(gs_base[0, 1]),
            ],
            [
                fig.add_subplot(gs_base[1, 0]),
                fig.add_subplot(gs_base[1, 1]),
            ],
        ]
    elif grid_type == "2x3":
        gs_base = fig.add_gridspec(
            nrows=2,
            ncols=3,
            left=0.07,
            right=0.98,
            bottom=0.09,
            top=0.94,
            wspace=0.3,
            hspace=0.3,
        )
        axes = [
            [
                fig.add_subplot(gs_base[0, 0]),
                fig.add_subplot(gs_base[0, 1]),
                fig.add_subplot(gs_base[0, 2]),
            ],
            [
                fig.add_subplot(gs_base[1, 0]),
                fig.add_subplot(gs_base[1, 1]),
                fig.add_subplot(gs_base[1, 2]),
            ],
        ]
    elif grid_type == "3x5":
        gs_base = fig.add_gridspec(
            nrows=3,
            ncols=5,
            left=0.07,
            right=0.98,
            bottom=0.09,
            top=0.94,
            wspace=0.3,
            hspace=0.3,
        )
        axes = [[fig.add_subplot(gs_base[i, j]) for j in range(5)] for i in range(3)]
    elif grid_type == "2x2-3":
        gs_base = fig.add_gridspec(
            nrows=1,
            ncols=2,
            width_ratios=(2, 3 + 1 / 8 + 1 / 16),
            left=0.075,
            right=0.98,
            bottom=0.095,
            top=0.915,
            wspace=0.175,
            hspace=0.5,
        )
        hspace = 0.3 if tight else 0.6
        gs_left = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=gs_base[0], wspace=0.4, hspace=hspace
        )

        gs_right = gridspec.GridSpecFromSubplotSpec(
            2, 3, subplot_spec=gs_base[1], wspace=0.4, hspace=hspace
        )
        axes = [
            [
                fig.add_subplot(gs_left[0, 0]),
                fig.add_subplot(gs_left[0, 1]),
                fig.add_subplot(gs_right[0, 0]),
                fig.add_subplot(gs_right[0, 1]),
                fig.add_subplot(gs_right[0, 2]),
            ],
            [
                fig.add_subplot(gs_left[1, 0]),
                fig.add_subplot(gs_left[1, 1]),
                fig.add_subplot(gs_right[1, 0]),
                fig.add_subplot(gs_right[1, 1]),
                fig.add_subplot(gs_right[1, 2]),
            ],
        ]
    elif grid_type == "2-3":
        gs_base = fig.add_gridspec(
            nrows=1,
            ncols=2,
            width_ratios=(2, 3 + 1 / 8 + 1 / 32),
            left=0.095,
            right=0.99,
            # bottom=0.085,
            bottom=0.18,
            # top=0.915,
            top=0.84,
            wspace=0.15,
            # wspace=0.15,
            hspace=0.5,
            # hspace=0.10,
        )
        gs_left = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs_base[0], wspace=0.4, hspace=0.7
        )
        gs_right = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=gs_base[1], wspace=0.4, hspace=0.7
        )
        axes = [
            [
                fig.add_subplot(gs_left[0, 0]),
                fig.add_subplot(gs_left[0, 1]),
                fig.add_subplot(gs_right[0, 0]),
                fig.add_subplot(gs_right[0, 1]),
                fig.add_subplot(gs_right[0, 2]),
            ]
        ]
    # print(
    #     f"Width of plots on the left and right should be equal. "
    #     + f"Currently ["
    #     + f"{(axes[0][0]._position.x1 - axes[0][0]._position.x0):.3f}"
    #     + ", "
    #     + f"{(axes[0][2]._position.x1 - axes[0][2]._position.x0):.3f}"
    #     + ")"
    # )
    return axes


def make_xticks_pow10(ax, xs):
    ax.set_xticks(xs)

    def format_pow(power):
        return "$\\mathdefault{10^" + str(int(power)) + "}$"

    labels = (
        [format_pow(np.log10(xs[0]))]
        + ["" for i in range(len(xs) - 2)]
        + [format_pow((np.log10(xs[-1])))]
    )

    ax.set_xticklabels(labels)
