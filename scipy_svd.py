import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
import os
from time import strftime

model = 'scipy_svd'
ordering = 'mu' # rows correspond to movie_ids; cols correspond to user_ids
submit = False # set to True to save a submission on qual


def three_dot(a, b, c):
    # Similar to dot product but for 3 equal length vectors
    return np.sum(a[i] * b[i] * c[i] for i in range(len(c)))


def RMSE(true, pred):
    return np.sqrt(np.sum((true - pred)**2) / len(true))


def predict(mean, U, s, Vt, row, col):
    # Evaluates predictions specified by SVD matrices U and V and 
    # singular values s. Row is a list of movie ids; Col is a list of user ids
    pred = []
    
    V = np.transpose(Vt)
    for r, c in zip(row, col):
        pred.append(mean + three_dot(U[r], s, V[c]))
    
    return np.array(pred)
    
    
def save_submission(pred):
    # Saves submission on qual set given predictions
    # File is saved as 'submissions/*.pred'
    filename = '_'.join([ordering, model, strftime('%b%d%H%M%S')]) + '.pred'
    f = open(os.path.join('submissions', filename), 'w')
    
    for p in pred:
        f.write('%.3f\n' % p)
        
    f.close()
    

print('Loading data...')
df = pd.read_csv(os.path.join('data', 'mu_train.csv'))

row = df['User Number'].values - 1
col = df['Movie Number'].values - 1
val = df['Rating'].values

df_val = pd.read_csv(os.path.join('data', 'mu_val.csv'))

row_val = df_val['User Number'].values - 1
col_val = df_val['Movie Number'].values - 1
val_val = df_val['Rating'].values


print('Solving SVD...')
dim = (max(df['User Number']), max(df['Movie Number']))

# mean center ratings to improve predictions, still shiet tho ¯\_(ツ)_/¯
mean = np.mean(val)
sparse_mat = coo_matrix((val - mean, (row, col)), shape=dim)

# U is N_users x k, s has length k, Vt is k x N_movies
U, s, Vt = svds(sparse_mat.asfptype(), k=20)

print('Train RMSE:', RMSE(val, predict(mean, U, s, Vt, row, col)))
print('Val RMSE:', RMSE(val_val, predict(mean, U, s, Vt, row_val, col_val)))


if submit:

    print('Saving submission...')
    df_qual = pd.read_csv(os.path.join('data', 'mu_qual.csv'))

    row_qual = df_qual['User Number'].values - 1
    col_qual = df_qual['Movie Number'].values - 1
    
    pred = predict(mean, U, s, Vt, row_qual, col_qual)
    save_submission(pred)