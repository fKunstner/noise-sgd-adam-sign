# Experiments 

Each folder contains a set of experiments in a file `experiment.py` 
containing the models, datasets, optimizers and grid of hyperparameters to try.

To run the experiments, install the `explib` library.
See [readme in the `explib` folder](../explib).


To generate scripts to run the experiments in an `experiment.py` file, 
use `python -m explib experiment.py`.
This should generate one `.json` file for each experiment and a submission file  
for use in `SLURM` clusters for submission.
```
explib_workspace/
└─ exp_name/
   ├─ exp_defs/
   │  ├─ ...
   │  ├─ exp_name_uuid1.json
   │  └─ exp_name_uuid2.json
   ├─ jobs/
   │  └─ main.sh
   └─ exp_name_summary.json
```

## Managing experiment logs

To centralize the results of the experiemnts, we use `wandb`.
To use the same setup, you will need a [`wandb`](https://wandb.ai/) instance.
See [readme in the `explib` folder](../explib)
for `explib`-related configuration.

After running the experiments, the logs should be in 
```
explib_workspace/
└─ exp_name/
   └─ logs/
      └─ wandb/
```
Unless running with `wandb` in online mode, 
the logs will need to be pushed manually.
Calling `wandb sync *` in the `logs/wandb` directory 
will push the results to the `wandb` instance.

Once the results are uploaded, 
they need to be validated with `python -m explib.results --checkup`.

If some experiments failed (out of memory or timeout or other issues)
a new `SLURM` file can be generated using 
`python -m explib experiment.py --unfinished`.
The `--unfinished` flag uses the results of 
`python -m explib.results --checkup`
to avoid re-running experiments that are marked as succesful in `wandb`
by the checkup.

## Downloading results

Once validated, the results can be downloaded to the `EXPLIB_WORKSPACE` folder with 
```
# download a list of all the runs stored on wandb
python -m explib.results --summary  

# download the detailled results for each run
python -m explib.results --download

# incremental version of the above
python -m explib.results --download-new
```

To prepare the results for the plotting code, use 
`python -m explib.results --concat`
to produce the concatenated file 
`EXPLIB_WORKSPACE/results/all_runs.csv`
used for plotting.
