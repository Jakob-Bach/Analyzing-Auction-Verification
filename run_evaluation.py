"""Run evaluation

Evaluation pipeline, creating plots for the paper and printing statistics which are used in the
paper as well. Should be run after the experimental pipeline.

Usage: python -m run_evaluation --help
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import prepare_dataset


# Run the full evaluation pipeline. To that end, read experiments' input files from "data_dir",
# experiments' results files from the "results_dir" and save plots to the "plot_dir".
# Print some statistics to the console.
def evaluate(data_dir: pathlib.Path, results_dir: pathlib.Path, plot_dir: pathlib.Path) -> None:
    if not plot_dir.is_dir():
        print('Plot directory does not exist. We create it.')
        plot_dir.mkdir(parents=True)
    if len(list(plot_dir.glob('*.pdf'))) > 0:
        print('Plot directory is not empty. Files might be overwritten, but not deleted.')

    dataset = prepare_dataset.load_dataset(data_dir=data_dir)
    results = pd.read_csv(results_dir / 'prediction_results.csv')

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

    corr_matrix_pearson = dataset.corr(method='pearson')
    print('\nPearson correlation of prices to revenue:')
    print(corr_matrix_pearson.loc[[f'allocation.p{i}.price' for i in range(1, 7)],
                                  'allocation.revenue'].sort_values().round(2))

    # --Verification result--

    print('\nHow often do different verification results occur?')
    print((dataset['verification.result'].value_counts() / len(dataset)).round(2))

    print('\nHow often are bidders verified positively as winners?')
    print(dataset.groupby('property.winner')['verification.result'].mean().round(2))

    print('\nHow often are products verified positively?')
    print(dataset.groupby('property.product')['verification.result'].mean().round(2))

    print('\nHow often are prices verified positively?')
    print(dataset.groupby('property.price')['verification.result'].mean())

    # --Verification time--

    print('\nHow is verification time (in s) distributed?')
    print((dataset['verification.time'] / 1000).describe().round(2))

    print('\nHow does verification time vary between product positions?')
    print(dataset.groupby('id.product_position')['verification.time'].describe().round().transpose())

    print('\nWhat is the average verification time per capacity of an individual bidder?')
    print(pd.concat([dataset.groupby(f'process.b{i}.capacity')['verification.time'].mean().rename_axis(
        'capacity').rename(f'b{i}') for i in range(1, 5)], axis='columns').fillna(0).round())

    # Figure 2
    plot_data = dataset[[f'process.b{i}.capacity' for i in range(1, 5)] + ['verification.time']].melt(
        id_vars='verification.time', value_vars=[f'process.b{i}.capacity' for i in range(1, 5)],
        value_name='Capacity', var_name='Bidder')
    plot_data['Bidder'] = plot_data['Bidder'].str.extract('process.b([0-9]).capacity').astype(int)
    plot_data['verification.time'] = plot_data['verification.time'] / 1000
    plt.figure(figsize=(4, 3))
    sns.boxplot(x='Bidder', y='verification.time', hue='Capacity', data=plot_data, fliersize=0,
                palette='OrRd')
    plt.ylabel('Verification time (seconds)')
    plt.legend(title='Capacity', edgecolor='white', loc='upper right', framealpha=0)
    plt.tight_layout()
    plt.savefig(plot_dir / 'time-vs-capacity.pdf')

    print('\nHow does mean verification time vary between products?')
    print(dataset.groupby('property.product')['verification.time'].describe().round().transpose())

    print('\nHow does mean verification time vary between bidders verified as winners?')
    print(dataset.groupby('property.winner')['verification.time'].describe().round().transpose())

    print('\nWhat is the average verification time per price?')
    print(dataset.groupby('property.price')['verification.time'].mean().round())

    print('\nHow does mean verification time vary between price and winner verification?')
    print(dataset.groupby(dataset['property.winner'].notna())['verification.time'].describe().transpose())

    print('\nCorrelation between verification result and verification time is',
          dataset['verification.result'].corr(dataset['verification.time'], method='pearson').round(2), '(Pearson),',
          dataset['verification.result'].corr(dataset['verification.time'], method='spearman').round(2), '(Spearman).')

    print('\nPearson correlation of features to verification time:')
    print(corr_matrix_pearson['verification.time'].sort_values().round(2))

    # ----Predicting the Data----

    # --Verification result--

    print('\nWhat is the prediction performance (MCC) for verification result?')
    print(results[results['target'] == 'verification.result'].groupby(['split_method', 'n_trees'])[
        ['train_score', 'test_score']].agg(['min', 'mean', 'median']).round(2))

    print('\nWhere is the test MCC for verification result lower than 0.9?')
    print(results.loc[(results['target'] == 'verification.result') & (results['test_score'] < 0.9),
                      ['fold_id', 'split_method', 'n_trees']])

    print('\nHow many unique feature (capacity + property) combinations are there?')
    prediction_features = [f'process.b{i}.capacity' for i in range(1, 4)] +\
        ['property.price', 'property.product', 'property.winner']
    print(dataset.fillna(0).groupby(prediction_features).ngroups)  # fillna() for empty winner

    print('\nHow many unique verification results are there per feature combination?')
    print(dataset.fillna(0).groupby(prediction_features)['verification.result'].nunique().value_counts())

    importance_cols = [x for x in results.columns if x.startswith(('imp_'))]
    print('\nHow does feature importance for verification result vary between and within the number of trees?')
    print(results[results['target'] == 'verification.result'].groupby('n_trees')[
        importance_cols].agg(['mean', 'std']).round(3).transpose().dropna())

    print('\nHow does feature importance for verification result vary between the split methods?')
    print(results[results['target'] == 'verification.result'].groupby('split_method')[
        importance_cols].mean().round(2).transpose().dropna())

    # Figure 3
    plot_data = results[(results['target'] == 'verification.result') & (results['n_trees'] == 100)].melt(
        value_vars=importance_cols, var_name='Feature', value_name='Importance').dropna()
    plot_data['Feature'] = plot_data['Feature'].str.replace('imp_(process\\.|property\\.)', '')
    plot_data = plot_data.groupby('Feature').mean().reset_index()  # variation too low for boxplot
    plt.figure(figsize=(4, 3))
    sns.barplot(x='Feature', y='Importance', data=plot_data, color='orange')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_dir / 'importance-result.pdf')


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
