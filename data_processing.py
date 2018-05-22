import pandas as pd
import numpy as np
import os

def movies_per_user(row, col, row_idx):
    '''
    Arguments:
        L - the first index in the dataset corresponding to each user in row_idx
        R - 1 + the last index in the dataset corresponding to each user in row_idx
        col - the full list of movies ids
        row_idx - the list of user ids we need the movies for
        
    Runtime:
        O(Sum |I_u|)
    
    Returns:
        A dictionary whose parameters are used to create a tf SparseTensor (2D)
    '''
    n_users = len(row_idx)
    
    L = row.searchsorted(row_idx, side='left')
    R = row.searchsorted(row_idx, side='right')
    
    ind_row = np.concatenate([np.full(R[u] - L[u], u) for u in range(n_users)])
    ind_col = np.concatenate([np.arange(R[u] - L[u]) for u in range(n_users)])
    val = np.concatenate([col[L[u]:R[u]] for u in range(n_users)])
    
    result = {
        'indices': np.transpose([ind_row, ind_col]), 
        'values': val,
        'dense_shape': (n_users, np.max(R - L))
    }
    
    return result