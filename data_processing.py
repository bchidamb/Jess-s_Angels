import pandas as pd
import numpy as np
import os

def movies_per_user(row, col, row_idx, n_movies_tot):
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
    
    out = np.full((n_users, np.max(R - L)), n_movies_tot)
    
    for u in range(n_users):
        out[u][:(R[u] - L[u])] = col[L[u]:R[u]]
    
    return out, (R - L)