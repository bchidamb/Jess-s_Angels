import pandas as pd
import numpy as np
import os

def movies_per_user(dataset):
    '''
    Arguments:
        df_um - A dataframe obtained by loading 'um_<dataset>.csv'
    
    Returns:
        A list of lists where the uth list is a list of movies rated by user u
        Note - movies are 0 indexed
    '''
    path = os.path.join('data', 'um_' + dataset + '.csv')
    df_um = pd.read_csv(path, dtype=np.int32)
    
    n_users = np.max(df_um['User Number'])
    users = list(1 + np.arange(n_users))

    L = df_um['User Number'].searchsorted(users, side='left')
    R = df_um['User Number'].searchsorted(users, side='right')
    
    ind_row = df_um['User Number'].values - 1
    ind_col = np.concatenate([np.arange(R[u] - L[u]) for u in range(n_users)])
    val = df_um['Movie Number'].values - 1
    
    result = {
        'indices': np.transpose([ind_row, ind_col]), 
        'values': val,
        'dense_shape': (n_users, np.max(R - L))
    }
    
    return result