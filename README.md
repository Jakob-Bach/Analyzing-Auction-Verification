# Analyzing and Predicting Verification of Data-Aware Process Models -- a Case Study with Spectrum Auctions

This repository contains the code of the paper

> Ordoni, Elaheh, Jakob Bach, and Ann-Katrin Hanke. "Analyzing and Predicting Verification of Data-Aware Process Models -- a Case Study with Spectrum Auctions"

published by [IEEE Access](https://www.doi.org/10.1109/ACCESS.2022.3154445) in 2022.

This document describes the steps to reproduce the experiments.
You can find the corresponding experimental data (inputs as well as results) on [KITopenData](https://doi.org/10.5445/IR/1000142949).

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
To install all necessary dependencies for this repo, simply run

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

## Reproducing the Experiments

After setting up and activating an environment, you are ready to run the code.
If you don't have access to the pre-processed dataset `auction_verification_large.csv`,
acquire the six original CSV files `result[0-5].csv` from the iterative verification procedure and run

```bash
python -m prepare_dataset
```

to pre-process the dataset.
Having obtained `auction_verification_large.csv`, start the prediction pipeline with

```bash
python -m run_experiments
```

To print statistics and create the plots for the paper, run

```bash
python -m run_evaluation
```

All scripts have a few command line options, which you can see by running the scripts like

```bash
python -m run_experiments --help
```

If you are fine with all input and output data to be stored in a directory called `data/`
(at your current location, e.g., in this repo), you can stick to the default arguments of the scripts.

## Creating the (Verification Result / Verification Time) Prediction Dataset

The prediction pipeline automatically creates two prediction datasets,
one for verification result and verification time, the other one for revenue.
If you want to create the verification result/time prediction dataset on a standalone basis,
obtain `auction_verification_large.csv` (as described above) and run

```bash
python -m create_prediction_dataset
```
