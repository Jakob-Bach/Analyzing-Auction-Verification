"""Create prediction dataset

Creates a standalone dataset to predict verification result or verification time with features for
- the auction design (i.e., bidders' capacities) in the data-aware process model
- the property to be verified (i.e., product, price, and winner)
This script is not necessary to reproduce the experiments.

Usage: python -m create_prediction_dataset --help
"""


import argparse
import pathlib

import prepare_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates a small result/time prediction dataset.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=pathlib.Path, default='data/', dest='data_dir',
                        help='Directory with input data, i.e., the original auction-verification' +
                        ' dataset, and to store output data, i.e., the created dataset.')
    data_dir = parser.parse_args().data_dir
    print('Dataset creation started.')
    if not data_dir.is_dir():
        print('Data directory does not exist. We create it.')
        data_dir.mkdir(parents=True)
    if any(data_dir.iterdir()):
        print('Data directory is not empty. Files might be overwritten, but not deleted.')
    dataset = prepare_dataset.load_dataset(data_dir=data_dir)
    dataset = prepare_dataset.create_deduplicated_dataset(dataset=dataset)
    dataset.to_csv(data_dir / 'auction_verification_prediction_data.csv', index=False)
    print('Dataset creation finished.')
