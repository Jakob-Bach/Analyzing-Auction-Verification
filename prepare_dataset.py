"""Prepare dataset

Data handling. Besides containing data-processing functions, also acts as a script to merge and
pre-process the results files from the verification algorithm.

Usage: python -m prepare_dataset --help
"""

import argparse
import pathlib

import pandas as pd


# Add various id features by modifying "dataset" in-place.
def add_id_features(dataset: pd.DataFrame) -> None:
    # Identify product permutations (all start with same capacity after winner for last product of
    # previous permutation was determined):
    dataset['id.product_permutation'] =\
        (dataset['property.winner'].isna() & dataset['property.winner'].shift(fill_value=0).notna() &
         (dataset['process.b1.capacity'] == 2) & (dataset['process.b2.capacity'] == 3) &
         (dataset['process.b3.capacity'] == 2) & (dataset['process.b4.capacity'] == 1)).cumsum().astype('Int64')
    # Identify iterations within each permutation setting:
    dataset['id.iteration'] = dataset.groupby('id.product_permutation').cumcount() + 1
    # Identify product position within each permutation setting (count changes of product):
    dataset['new_product'] = (dataset['property.product'] != dataset['property.product'].shift(fill_value=0)) |\
        (dataset['id.product_permutation'] != dataset['id.product_permutation'].shift(fill_value=0))
    dataset['id.product_position'] = dataset.groupby('id.product_permutation')['new_product'].cumsum()
    dataset.drop(columns='new_product', inplace=True)
    # Identify parallel cases for each product within a permutation (i.e., after finding winner,
    # verification restarts for same product, rather than moving on to next product):
    dataset['after_winner'] = dataset['property.winner'].isna() &\
        dataset['property.winner'].shift(fill_value=0).notna()
    dataset['id.product_case'] = dataset.groupby(
        ['id.product_permutation', 'property.product'])['after_winner'].cumsum()
    dataset.drop(columns='after_winner', inplace=True)


# Main routine for pre-processing the results files from the verification algorithm:
# - rename and re-order columns
# - adapt and extract some values
# - add id columns
def prepare_dataset(data_dir: pathlib.Path) -> None:
    if not data_dir.is_dir():
        print('Data directory does not exist. We create it.')
        data_dir.mkdir(parents=True)
    if any(data_dir.iterdir()):
        print('Data directory is not empty. Files might be overwritten, but not deleted.')
    dataset = pd.concat([pd.read_csv(x) for x in data_dir.glob('result*.csv')], ignore_index=True)
    # Use more intuitive column names, which also group columns logically:
    dataset.rename(columns={
        'property': 'property.formula', 'id': 'property.product', 'winner': 'property.winner',
        'lowestprice': 'property.price', 'capB1': 'process.b1.capacity', 'capB2': 'process.b2.capacity',
        'capB3': 'process.b3.capacity', 'capB4': 'process.b4.capacity', 'revenue': 'allocation.revenue',
        'last round': 'verification.is_final', 'result': 'verification.result', 'time': 'verification.time',
        'marking': 'verification.markings', 'edges': 'verification.edges'
    }, inplace=True)
    # Use NAs instead of zeros:
    dataset[['property.winner', 'allocation.revenue']] =\
        dataset[['property.winner', 'allocation.revenue']].astype('Int64').replace(0, float('nan'))
    # Extract numeric values out of final-allocation string:
    dataset['final allocation'].fillna('', inplace=True)  # for string extraction, need strings
    for i in range(1, 7):  # 6 products
        # why "[ ]?" -> in some datasets there is an additional whitespace in the string, in others not
        dataset[[f'allocation.p{i}.price', f'allocation.p{i}.winner']] = dataset['final allocation'].str.extract(
            f'product{i}: [ ]?price ([0-9]+) winner ([0-9])').astype(float).astype('Int64')
    dataset.drop(columns='final allocation', inplace=True)
    # Extract winner (was done partially, but not for rows with result == false, even if they are
    # winner-determination rows); note the binary encoding of the winner (digits for 1, 2, and 4):
    dataset.loc[dataset['property.formula'].str.contains('winner_1_1 > 0 AND winner_2_0 > 0 AND winner_4_0 > 0'),
                'property.winner'] = 1
    dataset.loc[dataset['property.formula'].str.contains('winner_1_0 > 0 AND winner_2_1 > 0 AND winner_4_0 > 0'),
                'property.winner'] = 2
    dataset.loc[dataset['property.formula'].str.contains('winner_1_1 > 0 AND winner_2_1 > 0 AND winner_4_0 > 0'),
                'property.winner'] = 3
    dataset.loc[dataset['property.formula'].str.contains('winner_1_0 > 0 AND winner_2_0 > 0 AND winner_4_1 > 0'),
                'property.winner'] = 4
    # Do not reduce capacity already in rows where winner is sucessfully determined (should only
    # be reduced afterwards, so we always have pre-verification capacity, independent from result):
    for bidder in range(1, 5):
        dataset[f'process.b{bidder}.capacity'] += dataset['verification.result'] &\
            (dataset['property.winner'] == bidder).fillna(False)
    # Add new identifier columns:
    add_id_features(dataset=dataset)
    # Re-order columns
    dataset = dataset[[x for x in dataset.columns if x.startswith('id')] +
                      [x for x in dataset.columns if x.startswith('process')] +
                      [x for x in dataset.columns if x.startswith('property')] +
                      [x for x in dataset.columns if x.startswith('verification')] +
                      [x for x in dataset.columns if x.startswith('allocation')]]
    dataset.to_csv(data_dir / 'auction_verification_large.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Merges and pre-processes the results files from the verification algorithm.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--directory', type=pathlib.Path, default='data/', dest='data_dir',
                        help='I/O directory for data (.csv files).')
    print('Dataset preparation started.')
    prepare_dataset(**vars(parser.parse_args()))
    print('Dataset prepared and saved.')
