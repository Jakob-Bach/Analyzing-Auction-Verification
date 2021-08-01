"""Run evaluation

Evaluation pipeline, creating plots for the paper and printing statistics which are used in the
paper as well. Should be run after the experimental pipeline.

Usage: python -m run_evaluation --help
"""

import argparse
import pathlib

import pandas as pd


# Run the full evaluation pipeline. To that end, read experiments' input files from "data_dir",
# experiments' results files from the "results_dir" and save plots to the "plot_dir".
# Print some statistics to the console.
def evaluate(data_dir: pathlib.Path, results_dir: pathlib.Path, plot_dir: pathlib.Path) -> None:
    if not plot_dir.is_dir():
        print('Plot directory does not exist. We create it.')
        plot_dir.mkdir(parents=True)
    if len(list(plot_dir.glob('*.pdf'))) > 0:
        print('Plot directory is not empty. Files might be overwritten, but not deleted.')

    dataset = pd.read_csv(data_dir / 'auction_verification_large.csv')
    float_cols = dataset.dtypes[dataset.dtypes == 'float'].index
    dataset[float_cols] = dataset[float_cols].astype('Int64')

    # ------Dataset------

    print('Number of rows:', dataset.shape[0])
    print('Number of features:', dataset.shape[1])

    # ------Evaluation------

    # ----Exploring the Data----

    # --Allocations--

    print('\nNumber of final allocations:', dataset['verification.is_final'].sum())
    print('Number of unique combinations of final prices:',
          dataset.groupby([f'allocation.p{i}.price' for i in range(1, 7)]).ngroups)
    print('Number of unique combinations of winner assignments:',
          dataset.groupby([f'allocation.p{i}.winner' for i in range(1, 7)]).ngroups)

    print('\nHow often does each final price occur for each product?')
    print(pd.concat([dataset[f'allocation.p{i}.price'].value_counts().rename_axis('price').rename(
        f'p{i}') for i in range(1, 7)], axis='columns').fillna(0).sort_index())

    print('\nHow often does each bidder win each product?')
    print(pd.concat([dataset[f'allocation.p{i}.winner'].value_counts().rename_axis('bidder').rename(
        f'p{i}') for i in range(1, 7)], axis='columns').fillna(0))

    print('\nHow often does each bidder acquire a certain number of products?')
    allocations = dataset[[f'allocation.p{i}.winner' for i in range(1, 7)]].value_counts().rename(
        'win_count').reset_index()
    allocations[[f'b{i}' for i in range(1, 5)]] = allocations.drop(columns='win_count').aggregate(
        pd.Series.value_counts, axis='columns').fillna(0).astype(int)  # num wins per bidder
    print(pd.concat([allocations.groupby(f'b{i}')['win_count'].sum().rename_axis('products_won').rename(
        f'b{i}') for i in range(1, 5)], axis='columns').fillna(0).astype(int))

    # --Revenue--

    print('\nHow often do different revenues occur?')
    print(dataset['allocation.revenue'].value_counts().sort_index())
    print(dataset['allocation.revenue'].describe().round(2))

    print('\nHow many distinct revenues are there per product permutation?')
    print(dataset.groupby('id.product_permutation')['allocation.revenue'].nunique().value_counts().sort_index())

    print('\nHow many distinct revenues are there per winner allocation?')
    print(dataset.groupby([f'allocation.p{i}.winner' for i in range(1, 7)])['allocation.revenue'].nunique(
        ).value_counts().sort_index())

    print('\nHow many distinct revenues are there per product-bidder allocation?')
    print(pd.concat([dataset.groupby(f'allocation.p{i}.winner')['allocation.revenue'].nunique().rename_axis(
        'bidder').rename(f'p{i}') for i in range(1, 7)], axis='columns').fillna(0).astype(int))

    print('\nHow many distinct revenues are there per product-price allocation?')
    print(pd.concat([dataset.groupby(f'allocation.p{i}.price')['allocation.revenue'].nunique().rename_axis(
        'price').rename(f'p{i}') for i in range(1, 7)], axis='columns').fillna(0).astype(int))


# Parse some command line argument and run evaluation.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates the paper\'s plots and prints statistics.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=pathlib.Path, default='data/', dest='data_dir',
                        help='Directory with input data, i.e., the auction-verification dataset.')
    parser.add_argument('-r', '--results', type=pathlib.Path, default='data/',
                        dest='results_dir', help='Directory with experimental results.')
    parser.add_argument('-p', '--plots', type=pathlib.Path, default='../plots/',
                        dest='plot_dir', help='Output directory for plots.')
    print('Evaluation started.')
    evaluate(**vars(parser.parse_args()))
    print('Plots created and saved.')
