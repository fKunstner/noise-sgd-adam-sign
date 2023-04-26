import os.path
import pickle as pk
from pathlib import Path

import explib.results.data as data_h
from explib import config
from explib.results import experiment_groups as expdef
from explib.results.cleanup import clean_data
from explib.results.data import get_all_runs, get_summary, gridsearch_for

CACHE_DIR = os.path.join(config.get_workspace(), "cache")

VERSION = 6


def cached_call(base_filename, function):
    filename = f"{base_filename}_{VERSION}.pk"
    filepath = os.path.join(CACHE_DIR, filename)
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

    if os.path.isfile(filepath):
        print(f"Loading {base_filename} from cache {filepath}")
        with open(filepath, "rb") as handle:
            return pk.load(handle)
    else:
        print(f"No cache  hit for {base_filename} ({function} at {filepath}). Build...")
        results = function()
        print(f"Saving {base_filename} from cache {filepath}")
        with open(filepath, "wb") as handle:
            pk.dump(results, handle, protocol=pk.HIGHEST_PROTOCOL)
        return results


def load_cleaned_data():
    def _load_cleaned_data():
        return clean_data(get_summary(), get_all_runs())

    return cached_call("cleaned_data", _load_cleaned_data)


def load_filtered_data():
    def _load_filtered_data():
        summary, runs = load_cleaned_data()

        summary_filter = data_h.flatten(
            [
                expdef.EXPERIMENTS[ds][bs]
                for ds in [
                    expdef.MNIST,
                    expdef.CIFAR10,
                    expdef.PTB,
                    expdef.WT2,
                    expdef.SQUAD,
                ]
                for bs in [expdef.S, expdef.M, expdef.L, expdef.XL, expdef.FULL]
            ]
        )

        return data_h.new_filter_and_merge(
            summary, runs, summary_filter, runs_filter=[]
        )

    return cached_call("filtered_data", _load_filtered_data)


def load_nd_filtered_data():
    def _load_nd_filtered_data():
        summary, runs = load_cleaned_data()

        summary_filter = data_h.flatten(
            [
                expdef.SPECIAL[expdef.NO_DROPOUT][ds][bs]
                for ds in [
                    expdef.PTB,
                    expdef.WT2,
                ]
                for bs in [expdef.FULL]
            ]
        )

        return data_h.new_filter_and_merge(
            summary, runs, summary_filter, runs_filter=[]
        )

    return cached_call("filtered_nd_data", _load_nd_filtered_data)


def gridsearch_nd_all_end():
    def _gridsearch_nd_all_end():
        data = load_nd_filtered_data()
        dss, bss, opts = [expdef.PTB, expdef.WT2], [expdef.FULL], expdef.ALL_MAIN_OPT
        return gridsearch_for(
            data,
            dss=dss,
            bss=bss,
            opts=opts,
            epoch_clip=expdef.EPOCH_CLIP,
            experiments=expdef.SPECIAL[expdef.NO_DROPOUT],
        )

    return cached_call("gridsearch_nd_all_end", _gridsearch_nd_all_end)


def gridsearch_all_end():
    def _gridsearch_all_end():
        data = load_filtered_data()
        dss, bss, opts = expdef.ALL_DS, expdef.ALL_BS, expdef.ALL_MAIN_OPT
        return gridsearch_for(
            data, dss=dss, bss=bss, opts=opts, epoch_clip=expdef.EPOCH_CLIP
        )

    return cached_call("gridsearch_all_end", _gridsearch_all_end)


def gridsearch_all_start():
    def _gridsearch_all_start():
        data = load_filtered_data()
        dss, bss, opts = expdef.ALL_DS, expdef.ALL_BS, expdef.ALL_MAIN_OPT
        return gridsearch_for(
            data, dss=dss, bss=bss, opts=opts, epoch_clip=expdef.EPOCH_CLIP_START
        )

    return cached_call("gridsearch_all_start", _gridsearch_all_start)


def gridsearch_all_start_soft_increase():
    def _gridsearch_all_start_soft_increase():
        data = load_filtered_data()
        dss, bss, opts = expdef.ALL_DS, expdef.ALL_BS, expdef.ALL_MAIN_OPT
        return gridsearch_for(
            data, dss=dss, bss=bss, opts=opts, epoch_clip=expdef.EPOCH_CLIP_START_NEW
        )

    return cached_call("gridsearch_all_start_new", _gridsearch_all_start_soft_increase)


def gridsearch_all_start_ignore_S():
    def _gridsearch_all_start_ignore_S():
        data = load_filtered_data()
        dss, bss, opts = expdef.ALL_DS, expdef.ALL_BS, expdef.ALL_MAIN_OPT
        return gridsearch_for(
            data,
            dss=dss,
            bss=bss,
            opts=opts,
            epoch_clip=expdef.EPOCH_CLIP_START_IGNORE_S,
        )

    return cached_call("gridsearch_all_ignore_S", _gridsearch_all_start_ignore_S)
