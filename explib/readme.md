# Explib

Instructions to install the `explib` library and set up an environment.

## Installing the library

Clone this repository, navigate to this folder. 
Install the requirements and the library.

This code was developped and tested using Pytorch `1.9.0`. 
To install this version, see https://pytorch.org/get-started/previous-versions/

```
pip install -r requirements.txt
pip install -r requirements-nocc.txt
pip install -e . 
```

## Set up environment to reproduce our figures 

The library expects the data to be stored in a specific folder, 
defined by an environment variable.
That directory is called `workspace` here, 
but the local name and location of the directory does not matter, 
it just needs to be specified by the environment variable `EXPLIB_WORKSPACE`.

On OSX/Unix: `export EXPLIB_WORKSPACE=~/workspace`   
On Windows: `set EXPLIB_WORKSPACE=C:\Users\user\path-to-workspace`  

## Download the data

To download the data, download `workspace.zip` 
from [the release page](https://github.com/fKunstner/noise-sgd-adam-sign/releases).  
Download and extract the content to the `EXPLIB_WORKSPACE` folder, to have this structure,
```
EXPLIB_WORKSPACE/
├─ norms_and_text_data/
└─ results/
   ├─ all_runs.csv
   └─ summary.csv
```
The `.zip` archive is compressed using `lzma` using `7zip`.  
On Windows, extract with [7-zip.org](www.7-zip.org).  
On OSX, extract with the default archive.   
On Unix, extract with `7z x workspace.zip` available from the package `p7zip-full`
(`apt` or `apt-get install p7zip-full`).

## Set up environment to run experiments 

Running the experiments requires a similar config as above. 
The dataset and results of the optimization will be stored in the `EXPLIB_WORKSPACE` directory. 
The library saves results to disk using wandb and can be used to upload the results to a wandb instance.

For Unix systems, use `export`. For Windows, use `set`
```
export EXPLIB_WORKSPACE=~/workspace
export WANDB_MODE=offline
export EXPLIB_WANDB_ENTITY=$wandb-entity
export EXPLIB_WANDB_PROJECT=$wandb-project
export EXPLIB_CONSOLE_LOGGING_LEVEL=INFO
```

The library can generate bash scripts to submit jobs to a SLURM cluster. 
To set up the SLURM account, use the environment variable `EXPLIB_SLURM_ACCOUNT=your-acc` 
