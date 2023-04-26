## Reproducing plots 

To reproduce the plots, first install `explib` library and download the data.
See [readme in the `explib` folder](../explib).

Run individual plots with `python script.py` or all with `run_all.sh`. 

The first call might take a while, but subsequent ones should be faster 
(intermediary results are cached).

To make a new plot or look at the raw data, use
```
from explib.results import data_caching
summary, runs = load_cleaned_data()
```
This loads the following files in as dataframes.
```
EXPLIB_WORKSPACE/results/summary.csv
EXPLIB_WORKSPACE/results/all_runs.csv
```
`summary` contains the metadata about the runs (optimizer/model/hyperparameters)
and `runs` contain the logs of each run.
