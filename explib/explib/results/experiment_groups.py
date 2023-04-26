"""Definition of "groups" of experiments mapping (dataset x batch size) to keys
to be selected from the experiment dataframe.

Simplified from explib/results/plotting.py
"""

##
# Dataset names

MNIST = "mnist"
WT2 = "wikitext2"
PTB = "ptb"
CIFAR10 = "cifar10"
SQUAD = "squad"
ALL_DS = [MNIST, CIFAR10, PTB, WT2, SQUAD]
##
# Batch size names

S = "S"
M = "M"
L = "L"
XL = "XL"
FULL = "Full"
ALL_BS = [S, M, L, XL, FULL]

##
# Experiment group names

NO_DROPOUT = "no-dropout"
NORM_ABLATION_FULL = "full-batch-training-normalized-optimizers"
FULL_BATCH = "full-batch-training"
INCREASING_BATCH_SIZE = "increasing-batch-size"
NORM_ABLATION = "normalization-ablation"

##
# Optimizer names

SGD_ANY = "SGD"
SGD_NM = "sgd-m"
SGD_M = "sgd+m"
ADAM_NM = "adam-m"
ADAM_M = "adam+m"
SIGN_NM = "SignDescent"
SIGN_M = "SignDescent+m"
BLOCK_NM = "BlockNormalizedGD"
BLOCK_M = "BlockNormalizedGD+m"
NORM_NM = "NormalizedGD"
NORM_M = "NormalizedGD+m"
RSD_NM = "RescaledSignDescent"
RSD_M = "RescaledSignDescent+m"
STANDARD_OPT = [SGD_M, SGD_NM, ADAM_M, ADAM_NM]
NORMALIZED_OPT = [
    SIGN_M,
    SIGN_NM,
    NORM_NM,
    NORM_M,
]
ALL_MAIN_OPT = STANDARD_OPT + NORMALIZED_OPT

##
# Helper functions to generate selection dictionaries


def no_dropout(ds, max, batch, acc):
    return {
        "group": NO_DROPOUT,
        "dataset": ds,
        "max_epoch": max,
        "accumulate_steps": acc,
        "batch_size": batch,
    }


def norm_abl(ds, max, batch, acc=None):
    acc_dict = {}
    if acc is not None:
        acc_dict["accumulate_steps"] = acc
    return {
        "group": NORM_ABLATION,
        "dataset": ds,
        "max_epoch": max,
        "batch_size": batch,
        **acc_dict,
    }


def incr(ds, batch, max, acc=None):
    acc_dict = {}
    if acc is not None:
        acc_dict["accumulate_steps"] = acc
    return {
        "group": INCREASING_BATCH_SIZE,
        "dataset": ds,
        "batch_size": batch,
        "max_epoch": max,
        **acc_dict,
    }


def full(ds):
    return {"group": FULL_BATCH, "dataset": ds}


def full_norm(ds, max):
    return {"group": NORM_ABLATION_FULL, "dataset": ds, "max_epoch": max}


##
# Main experiment groups
#
# Experimental settings used for the main experiment

EXPERIMENTS = {
    MNIST: {
        S: [
            norm_abl(ds=MNIST, batch=256, max=100),
            incr(ds=MNIST, batch=256, max=100),
        ],
        M: [
            norm_abl(ds=MNIST, max=100, batch=1024),
            incr(ds=MNIST, max=100, batch=1024),
        ],
        L: [
            norm_abl(ds=MNIST, max=200, batch=4096),
            incr(ds=MNIST, max=200, batch=4096),
        ],
        XL: [
            norm_abl(ds=MNIST, max=800, batch=16384),
            incr(ds=MNIST, max=800, batch=16384),
        ],
        FULL: [
            full(ds=MNIST),
            full_norm(ds=MNIST, max=800),
        ],
    },
    CIFAR10: {
        S: [
            incr(ds=CIFAR10, max=100, batch=64),
            norm_abl(ds=CIFAR10, max=100, batch=64),
        ],
        M: [
            incr(ds=CIFAR10, max=100, batch=256),
            norm_abl(ds=CIFAR10, max=100, batch=256),
        ],
        L: [
            incr(ds=CIFAR10, max=200, batch=1024),
            norm_abl(ds=CIFAR10, max=200, batch=1024),
        ],
        XL: [
            incr(ds=CIFAR10, max=800, batch=4096),
            norm_abl(ds=CIFAR10, max=400, batch=4096),
        ],
        FULL: [
            full_norm(ds=CIFAR10, max=800),
            full(ds=CIFAR10),
        ],
    },
    PTB: {
        S: [
            incr(ds=PTB, max=100, batch=16),
            norm_abl(ds=PTB, max=100, batch=16),
        ],
        M: [
            incr(ds=PTB, max=100, batch=64),
            norm_abl(ds=PTB, max=100, batch=64),
        ],
        L: [
            incr(ds=PTB, max=200, batch=256),
            norm_abl(ds=PTB, max=200, batch=256),
        ],
        XL: [
            incr(ds=PTB, max=800, batch=1024),
            norm_abl(ds=PTB, max=800, batch=1024),
        ],
        FULL: [
            full_norm(ds=PTB, max=3200),
            full(ds=PTB),
        ],
    },
    WT2: {
        S: [
            incr(ds=WT2, batch=20, max=40, acc=1),
            norm_abl(ds=WT2, batch=20, max=40, acc=1),
        ],
        M: [
            incr(ds=WT2, batch=80, max=40, acc=1),
            norm_abl(ds=WT2, batch=80, max=40, acc=1),
        ],
        L: [
            incr(ds=WT2, batch=80, max=80, acc=4),
            norm_abl(ds=WT2, batch=80, max=80, acc=4),
        ],
        XL: [
            incr(ds=WT2, batch=80, max=160, acc=16),
            norm_abl(ds=WT2, batch=80, max=160, acc=16),
        ],
        FULL: [
            full(ds=WT2),
            full_norm(ds=WT2, max=320),
        ],
    },
    SQUAD: {
        S: [
            incr(ds=SQUAD, batch=32, max=5, acc=1),
            norm_abl(ds=SQUAD, max=10, batch=16, acc=2),
        ],
        M: [
            incr(ds=SQUAD, batch=32, max=5, acc=4),
            norm_abl(ds=SQUAD, max=10, batch=16, acc=8),
        ],
        L: [
            incr(ds=SQUAD, batch=32, max=5, acc=16),
            norm_abl(ds=SQUAD, max=10, batch=16, acc=32),
        ],
        XL: [
            incr(ds=SQUAD, batch=32, max=20, acc=64),
            norm_abl(ds=SQUAD, max=10, batch=16, acc=128),
        ],
        FULL: [
            {
                "group": "fix-full-batch-training-squad",
                "max_epoch": 80,
                "batch_size": 16,
                "accumulate_steps": 1370 * 4,
            }
        ],
    },
}

EFF_BS = {
    CIFAR10: {S: 64, M: 256, L: 1024, XL: 4096, FULL: 50000},
    MNIST: {S: 256, M: 1024, L: 4096, XL: 16384, FULL: 60000},
    PTB: {S: 16, M: 64, L: 256, XL: 1024, FULL: 26520},
    WT2: {S: 20, M: 80, L: 320, XL: 1280, FULL: 16240},
    SQUAD: {S: 32, M: 128, L: 512, XL: 2048, FULL: 87680},
}

##
# Sanity check experiments

SPECIAL = {
    NO_DROPOUT: {
        PTB: {FULL: [no_dropout(ds=PTB, max=3200, acc=20, batch=1326)]},
        WT2: {FULL: [no_dropout(ds=WT2, max=320, batch=80, acc=203)]},
    }
}

##
# At what epoch to truncate the runs

EPOCH_CLIP = {
    MNIST: {S: 100, M: 100, L: 200, XL: 400, FULL: 800},
    CIFAR10: {S: 100, M: 100, L: 100, XL: 200, FULL: 800},
    PTB: {S: 100, M: 100, L: 200, XL: 400, FULL: 3200},
    WT2: {S: 40, M: 40, L: 40, XL: 80, FULL: 320},
    SQUAD: {S: 5, M: 5, L: 5, XL: 8, FULL: 60},
}

EPOCH_CLIP_START = {
    MNIST: {S: 4, M: 14, L: 58, XL: 267, FULL: 800},
    CIFAR10: {S: 1, M: 4, L: 16, XL: 64, FULL: 768},
    PTB: {S: 2, M: 8, L: 32, XL: 128, FULL: 3200},
    WT2: {S: 1, M: 2, L: 7, XL: 27, FULL: 320},
    SQUAD: {S: 1, M: 1, L: 1, XL: 2, FULL: 60},
}

EPOCH_CLIP_START_NEW = {
    MNIST: {S: 4, M: 14, L: 58, XL: 267, FULL: 800},
    CIFAR10: {S: 1, M: 4, L: 16, XL: 64, FULL: 768},
    PTB: {S: 2, M: 8, L: 32, XL: 128, FULL: 3200},
    WT2: {S: 1, M: 3, L: 10, XL: 34, FULL: 320},
    SQUAD: {S: 1, M: 2, L: 3, XL: 4, FULL: 60},
}

EPOCH_CLIP_START_IGNORE_S = {
    MNIST: {S: 4, M: 14, L: 58, XL: 267, FULL: 800},
    CIFAR10: {S: 1, M: 4, L: 16, XL: 64, FULL: 768},
    PTB: {S: 2, M: 8, L: 32, XL: 128, FULL: 3200},
    WT2: {S: 1, M: 2, L: 7, XL: 28, FULL: 320},
    SQUAD: {S: 1, M: 1, L: 1, XL: 2, FULL: 60},
}

##
# Named optimizer selections

OPTIMS = {
    SGD_ANY: {"opt.name": "SGD"},
    SGD_NM: {"opt.name": "SGD", "opt.momentum": 0.0},
    SGD_M: {"opt.name": "SGD", "opt.momentum": 0.9},
    ADAM_NM: {"opt.name": "Adam", "opt.b1": 0.0},
    ADAM_M: {"opt.name": "Adam", "opt.b1": 0.9},
    SIGN_NM: {"opt.name": "SignDescent", "opt.momentum": 0.0},
    SIGN_M: {"opt.name": "SignDescent", "opt.momentum": 0.9},
    NORM_NM: {"opt.name": "NormalizedGD", "opt.momentum": 0.0},
    NORM_M: {"opt.name": "NormalizedGD", "opt.momentum": 0.9},
    BLOCK_NM: {"opt.name": "BlockNormalizedGD", "opt.momentum": 0.0},
    BLOCK_M: {"opt.name": "BlockNormalizedGD", "opt.momentum": 0.9},
    RSD_NM: {"opt.name": "RescaledSignDescent", "opt.momentum": 0.0, "opt.norm": 1.0},
    RSD_M: {"opt.name": "RescaledSignDescent", "opt.momentum": 0.9, "opt.norm": 1.0},
}
