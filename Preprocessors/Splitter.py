import numpy as np

""" """

import numpy as np
import pandas as pd



def split_train_test(filenames, fraction=0.75, seed=None):
    """ Returns the list splited in train and test lists.
            · Filenames: pandas dataframe with the filenames innit
        Returns:
            · Dict containing two pandas datasets
    """

    n_files = filenames.shape[0]
    train_ids = np.random.choice(n_files, int(n_files * fraction),
                                 replace=False) 
    train = filenames.loc[train_ids, ]
    test = filenames.drop(train_ids)
    return {'train':train, 'test':test}

