"""Run experiments

Main experimental pipeline. Trains prediction models for different targets with different feature
sets, using models of different complexity and different cross-validation splits. Saves the
results as a CSV.

Usage: python -m run_experiments --help
"""

import argparse
import multiprocessing
import pathlib
import time
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
import sklearn.ensemble
import sklearn.model_selection
import tqdm

import prepare_dataset


NUM_TREES = [100, 10, 1]  # different model complexities; reverse order for better load balancing


def split_capacity(dataset: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    return sklearn.model_selection.LeaveOneGroupOut().split(
        X=dataset, groups=dataset.groupby([x for x in dataset.columns if 'capacity' in x]).ngroup())


def split_kfold(dataset: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    return sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=25).split(X=dataset)


def split_position(dataset: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    return sklearn.model_selection.LeaveOneGroupOut().split(X=dataset, groups=dataset['order.p1.pos'])


def split_product(dataset: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    return sklearn.model_selection.LeaveOneGroupOut().split(X=dataset, groups=dataset['property.product'])


def split_reverse_kfold(dataset: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    for train_idx, test_idx in split_kfold(dataset=dataset):
        yield test_idx, train_idx  # use smaller part for training, larger for testing


SPLIT_FUNCTIONS = [split_kfold, split_reverse_kfold, split_capacity, split_product]  # to evaluate generalization
SPLIT_FUNCTIONS_REVENUE = [split_kfold, split_reverse_kfold, split_position]  # different features, so different splits


# Define the tasks for the experimental pipeline, which are combinations of:
# - datasets (target + features)
# - split methods
# - model complexities (number of trees)
def define_experimental_design(dataset: pd.DataFrame) -> List[Dict[str, Any]]:
    results = []
    basic_dataset = prepare_dataset.create_deduplicated_dataset(dataset=dataset)
    basic_features = [x for x in basic_dataset.columns if x not in ['verification.result', 'verification.time']]
    revenue_dataset = prepare_dataset.create_revenue_dataset(dataset=dataset)
    revenue_features = [x for x in revenue_dataset.columns if x != 'allocation.revenue']
    for n_trees in NUM_TREES:
        for split_func in SPLIT_FUNCTIONS:
            results.append({'target': 'verification.result', 'features': basic_features,
                            'dataset': basic_dataset, 'split_func': split_func, 'n_trees': n_trees})
            results.append({'target': 'verification.time', 'features': basic_features,
                            'dataset': basic_dataset, 'split_func': split_func, 'n_trees': n_trees})
        for split_func in SPLIT_FUNCTIONS_REVENUE:
            results.append({'target': 'allocation.revenue', 'features': revenue_features,
                            'dataset': revenue_dataset, 'split_func': split_func, 'n_trees': n_trees})
    for i, result in enumerate(results):
        result['task_id'] = i
    return results


# Evaluate predictions for one "dataset", i.e., use its "features" and "target" column(s) to train
# a random forest (classification or regression) with "n_trees". Use the "split_func" to generate
# cross-validation splits. "task_id" is just copied to the output.
# Return a data frame with prediction performance, feature importance, and columns identifying the
# experimental setting.
def train_and_evaluate(dataset: pd.DataFrame, target: str, features: List[str],
                       split_func: Callable, n_trees: int, task_id: int) -> pd.DataFrame:
    prediction_results = []
    feature_importances = []
    for fold_id, (train_idx, test_idx) in enumerate(split_func(dataset=dataset)):
        X_train = dataset.loc[train_idx, features]
        y_train = dataset.loc[train_idx, target].astype(int)  # "verification.result" is bool
        X_test = dataset.loc[test_idx, features]
        y_test = dataset.loc[test_idx, target].astype(int)
        if dataset[target].nunique() == 2:
            model = sklearn.ensemble.RandomForestClassifier(n_estimators=n_trees, random_state=25)
            scoring_func = sklearn.metrics.matthews_corrcoef
        else:
            model = sklearn.ensemble.RandomForestRegressor(n_estimators=n_trees, random_state=25)
            scoring_func = sklearn.metrics.r2_score
        start_time = time.process_time()
        model.fit(X_train, y_train)
        end_time = time.process_time()
        train_score = scoring_func(y_true=y_train, y_pred=model.predict(X_train))
        test_score = scoring_func(y_true=y_test, y_pred=model.predict(X_test))
        result = {'fold_id': fold_id, 'train_score': train_score, 'test_score': test_score,
                  'training_time': end_time - start_time}
        prediction_results.append(result)
        feature_importances.append(model.feature_importances_)
    prediction_results = pd.DataFrame(prediction_results)
    prediction_results['task_id'] = task_id
    prediction_results['target'] = target
    prediction_results['split_method'] = split_func.__name__.replace('split_', '')
    prediction_results['n_trees'] = n_trees
    feature_importances = pd.DataFrame(feature_importances, columns=['imp_' + x for x in features])
    prediction_results = pd.concat([prediction_results, feature_importances], axis='columns')
    return prediction_results


# Run all experiments and save results.
def run_experiments(data_dir: pathlib.Path, results_dir: pathlib.Path,
                    n_processes: Optional[int] = None) -> None:
    if not data_dir.is_dir():
        raise FileNotFoundError('Data directory does not exist.')
    if not results_dir.is_dir():
        print('Results directory does not exist. We create it.')
        results_dir.mkdir(parents=True)
    if any(results_dir.iterdir()):
        print('Results directory is not empty. Files might be overwritten, but not deleted.')
    dataset = prepare_dataset.load_dataset(data_dir=data_dir)
    task_list = define_experimental_design(dataset=dataset)
    print('Running evaluation...')
    progress_bar = tqdm.tqdm(total=len(task_list))
    process_pool = multiprocessing.Pool(processes=n_processes)
    results = [process_pool.apply_async(train_and_evaluate, kwds=task,
                                        callback=lambda x: progress_bar.update())
               for task in task_list]
    process_pool.close()
    process_pool.join()
    progress_bar.close()
    pd.concat([x.get() for x in results]).to_csv(results_dir / 'prediction_results.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs the experimental pipeline.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=pathlib.Path, default='data/', dest='data_dir',
                        help='Directory with input data, i.e., the auction-verification dataset.')
    parser.add_argument('-r', '--results', type=pathlib.Path, default='data/', dest='results_dir',
                        help='Directory for output data, i.e., experimental results.')
    parser.add_argument('-p', '--processes', type=int, default=None, dest='n_processes',
                        help='Number of processes for multi-processing (default: all cores).')
    print('Experimental pipeline started.')
    run_experiments(**vars(parser.parse_args()))
    print('Experimental pipeline executed successfully.')
