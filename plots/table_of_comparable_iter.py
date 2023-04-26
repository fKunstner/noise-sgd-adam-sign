import cmd

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
    do_end = False

    importlib.reload(data_cache)

    if do_end:
        runs_at_last_epoch, best_runs = data_cache.gridsearch_all_end()
    else:
        runs_at_last_epoch, best_runs = data_cache.gridsearch_all_start_soft_increase()

    data = [runs_at_last_epoch, best_runs]

    for i in range(len(data)):
        data[i] = data_h.add_stop_at_info(data[i], stop_at=expdef.EPOCH_CLIP_START_NEW)

    return data


def postprocess(data):
    return data


def settings(plt):
    plt.rcParams.update(
        plth.smaller_config(
            nrows=1,
            ncols=1,
            height_to_width_ratio=1 / 1,
            rel_width=0.1,
        ),
    )
    pass


def make_figure(fig, data):
    def make_table_of_iterations():
        subset = data[0][
            [
                "dataset",
                "max_epoch",
                "grad_updates_per_epoch",
                "epoch_to_stop",
                "eff_bs",
            ]
        ]

        subset = subset.drop_duplicates()
        subset = subset.sort_values(by=["dataset", "eff_bs"])
        subset["updates_at_end"] = (
            subset["grad_updates_per_epoch"] * subset["epoch_to_stop"]
        )

        print(data)

        def makeline(items):
            return " & ".join([f"{x:<30}" for x in items]) + r" \\"

        print(r"\begin{tabular}{lrllll}")
        print(r"\toprule")
        print(
            makeline(
                [
                    "Dataset",
                    r"\multicolumn{2}{l}{Batch size}",
                    r"\# Epochs",
                    r"\# Iterations",
                ]
            )
        )
        print(r"\midrule")
        for ds in expdef.ALL_DS:
            for i, bs in enumerate(expdef.ALL_BS):
                data_ = data_h.new_select(
                    subset,
                    selections=[{"dataset": ds, "eff_bs": expdef.EFF_BS[ds][bs]}],
                ).to_dict("records")[0]
                if i == 0:
                    print(
                        makeline(
                            [
                                data_["dataset"],
                                f"{bs:>10} & {data_['eff_bs']:<10}",
                                data_["epoch_to_stop"],
                                data_["updates_at_end"],
                            ]
                        )
                    )
                else:
                    print(
                        makeline(
                            [
                                "",
                                f"{bs:>10} & {data_['eff_bs']:<10}",
                                data_["epoch_to_stop"],
                                data_["updates_at_end"],
                            ]
                        )
                    )
        print(r"\bottomrule")
        print(r"\end{tabular}")

    make_table_of_iterations()


if __name__ == "__main__":
    make_figure(plt.figure(), load_data())
    plt.show()
