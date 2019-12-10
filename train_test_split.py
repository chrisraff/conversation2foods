'''
Specify which documents from 'bert_evidence' to use for a test set,
and use the rest for training. Put all the vectors in to a training
csv and a testing csv. The training set can include the augmented
data from bert_evidence_augmented'
'''
from pathlib import Path
from tqdm import tqdm
import pandas as pd


bert_evidence = Path('bert_evidence/')
bert_evidence_augmented = Path('bert_evidence_augmented/')


# returns (training, testing) dataframes
# `test_fnames` is an array of strings that are the transcripts to be used for testing
# if `use_augmented` is true, the augmented data will be put in the training set
# the training set is balanced if `balanced` is true
def build_dataset(test_fnames, use_augmented=True, balanced=True):
    pd_test = None
    pd_train = None

    for f_path in tqdm(list(bert_evidence.glob('*.csv'))):
        f_name = f_path.parts[-1]

        if f_name in test_fnames:
            df_new = pd.read_csv(bert_evidence / f_name, index_col=0)

            if pd_test is None:
                pd_test = df_new
            else:
                pd_test = pd_test.append(df_new)

        else:
            # training set
            df_new = pd.read_csv(bert_evidence / f_name, index_col=0)

            if use_augmented:
                df_new_augmented = pd.read_csv(bert_evidence_augmented / f_name, index_col=0)

                df_new = df_new.append(df_new_augmented)
                df_new.reset_index(drop=True, inplace=True)

            if balanced:
                unique_labels = df_new.labels.value_counts()

                lowest_count = unique_labels.min()

                for label in unique_labels.keys():
                    # find n random points with this label and remove them
                    excess_amount = unique_labels[label] - lowest_count

                    this_label_df = df_new.loc[df_new['labels'] == label]

                    excess_points = this_label_df.sample(n=excess_amount)

                    assert len(excess_points) == excess_amount, f'got {len(excess_points)} points but should\'ve got {excess_amount}'
                    assert len(df_new.index) == len(df_new.index.unique()), f'unique idxs: {len(df_new.index.unique())}, expected {len(df_new.index)}'

                    df_new = df_new.drop(excess_points.index)

                    assert len(df_new.loc[df_new['labels'] == label]) == lowest_count, f"reduced to {len(df_new.loc[df_new['labels'] == label])} points but should\'ve been {lowest_count}"
            
            if pd_train is None:
                pd_train = df_new
            else:
                pd_train = pd_train.append(df_new)

    return pd_train, pd_test

if __name__ == '__main__':

    output_fname_prefix = 'bert_evidence_' # appended with 'train' and 'test'

    test_fnames = [
        'admmt7.csv',
        'inamt2.csv',
        'jammt5.csv',
        'kurtm1.csv',
        'kurmt2.csv',
        'kurmt5.csv',
        'megmt1.csv',
        'megmt2.csv',
        'mrkmt5.csv',
        'melmt5.csv'
    ]

    df_train, df_test = build_dataset(test_fnames)

    print('saving datasets')
    df_train.to_csv(output_fname_prefix + 'train.csv')
    df_test.to_csv(output_fname_prefix + 'test.csv')
