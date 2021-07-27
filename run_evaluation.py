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
    print(f'The dataset contains {dataset.shape[0]} data objects and {dataset.shape[1]} features.')


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
