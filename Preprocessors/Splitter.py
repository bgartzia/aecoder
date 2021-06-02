""" """

import numpy as np
import pandas as pd


def split_OK_NOK(anotations_file_path):

    files_df = pd.read_csv(anotations_file_path)
    files_df['FileName'] = files_df['FileName'] + '_' + files_df['part']
    ok = files_df[files_df['estado'] == 'OK']['FileName']
    nok = files_df[files_df['estado'] != 'OK']['FileName']

    return ok, nok
    


def split_train_test(filenames, fraction=0.75, seed=None):
    """ Returns the list splited in train and test lists.
            · Filenames: pandas dataframe with the filenames innit
        Returns:
            · Dict containing two pandas datasets
    """

    n_files = filenames.shape[0]
    np.random.seed(seed)
    train_ids = np.random.choice(n_files, int(n_files * fraction),
                                 replace=False) 
    train = filenames.iloc[train_ids]
    test = filenames.drop(filenames.index[train_ids])
    return {'train':train, 'test':test}

