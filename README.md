# Analyzing and Predicting Verification of Process Models -- A Case Study on Spectrum Auctions

This repository contains the code and text of the paper

> Ordoni, Elaheh, Jakob Bach, and Ann-Katrin Hanke. "Analyzing and Predicting Verification of Process Models -- A Case Study on Spectrum Auctions"

This document describes the steps to reproduce the experiments.

## Setup

Before running scripts to reproduce the experiments, you need to set up an environment with all necessary dependencies.
Our code is implemented in Python (version 3.8; other versions, including lower ones, might work as well).

### Option 1: `conda` Environment

If you use `conda`, you can install the right Python version into a new `conda` environment
and activate the environment as follows:

```bash
conda create --name <conda-env-name> python=3.8
conda activate <conda-env-name>
```

### Option 2: `virtualenv` Environment

We used [`virtualenv`](https://virtualenv.pypa.io/) (version 20.4.7; other versions might work as well) to create an environment for our experiments.
First, make sure you have the right Python version available.
Next, you can install `virtualenv` with

```bash
python -m pip install virtualenv==20.4.7
```

To set up an environment with `virtualenv`, run

```bash
python -m virtualenv -p <path/to/right/python/executable> <path/to/env/destination>
```

Activate the environment in Linux with

```bash
source <path/to/env/destination>/bin/activate
```

Activate the environment in Windows (note the back-slashes) with

```cmd
<path\to\env\destination>\Scripts\activate
```

### Dependency Management

After activating the environment, you can use `python` and `pip` as usual.
To install all necessary dependencies for this repo, switch to the directory `code` and simply run

```bash
python -m pip install -r requirements.txt
```

If you make changes to the environment and you want to persist them, run

```bash
python -m pip freeze > requirements.txt
```

To leave the environment, run

```bash
deactivate
```

### Optional Dependencies

To use the environment in the IDE `Spyder`, you need to install `spyder-kernels` into the environment.

## Reproducing the Experiments

After setting up and activating an environment, you are ready to run the code.
If you don't have access to the pre-processed dataset `auction_verification_large.csv`, acquire the raw CSV files and run

```bash
python -m prepare_dataset
```

to pre-process the dataset.
Having obtained `auction_verification_large.csv`, start the pipeline with

```bash
python -m run_experiments
```

To create the plots for the paper, run

```bash
python -m run_evaluation
```

All scripts have a few command line options, which you can see by running the scripts like

```bash
python -m run_experiments --help
```
