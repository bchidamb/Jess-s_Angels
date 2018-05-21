import tensorflow as tf
from tensorflow_utils import *
import numpy as np
import pandas as pd
import os
import sys
from time import clock
from data_processing import *
from utils import *
import gc

submit = False
model_name = 'tensorflow_svd++'
ordering = 'mu'


# useful links:
# https://github.com/aymericdamien/TensorFlow-Examples
# https://github.com/songgc/TF-recomm
# http://surprise.readthedocs.io/en/stable/matrix_factorization.html

# used techniques from this tensorflow implementation of SVD++:
# https://github.com/WindQAQ/tf-recsys/blob/master/tfcf/models/svdpp.py


def SVDpp(n_samples, n_u, n_m, mean, mpu, lf=100, reg=0.02, learning_rate=0.005):

    i = tf.placeholder(tf.int32, shape=[None])
    j = tf.placeholder(tf.int32, shape=[None])
    r = tf.placeholder(tf.float32, shape=[None])
    
    batch = tf.shape(i)[0]
    
    inits = tf.random_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float16)
    regs = tf.contrib.layers.l2_regularizer(scale=reg)
    
    mu = tf.constant([mean])
    b_u = tf.get_variable('user_bias', shape=[n_u], initializer=inits, regularizer=regs)
    b_m = tf.get_variable('movie_bias', shape=[n_m], initializer=inits, regularizer=regs)
    p_u = tf.get_variable('user_embedding', shape=[n_u, lf], initializer=inits, regularizer=regs)
    q_m = tf.get_variable('movie_embedding', shape=[n_m, lf], initializer=inits, regularizer=regs)
    y_m = tf.get_variable('implicit_movie_embedding', shape=[n_m, lf], initializer=inits, regularizer=regs)
    
    mpu_inds = tf.SparseTensor(mpu['indices'], mpu['values'], mpu['dense_shape'])
    I_u = tf.nn.embedding_lookup_sparse(y_m, mpu_inds, sp_weights=None, combiner='sqrtn')
    
    slice = tf.nn.embedding_lookup
    
    # prediction is u + b_u + b_i + q_m . (p_u + sum_(j in I_u) (y_j) / |I_u|^0.5)
    r_pred = tf.tile(mu, [batch]) \
        + slice(b_u, i) \
        + slice(b_m, j) \
        + tf.reduce_sum(tf.multiply(slice(p_u, i) + slice(I_u, i), slice(q_m, j)), 1)
        
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    train_loss = tf.reduce_sum(tf.pow(r_pred-r,2)) + sum(reg_losses)
    se = tf.reduce_sum(tf.pow(r_pred-r,2))
    model = tf.train.AdamOptimizer(learning_rate).minimize(train_loss)
    
    return i, j, r, se, r_pred, model
    

print('Loading data...')

train_dataset='train'

mpu = movies_per_user(train_dataset)

col_types = {'User Number': np.int32, 'Movie Number': np.int16, 'Rating': np.int8}
df = pd.read_csv(os.path.join('data', 'mu_' + train_dataset + '.csv'), dtype=col_types)

row = df['User Number'].values - 1
col = df['Movie Number'].values - 1
val = df['Rating'].values
print(sys.getsizeof(row))

n_samples = len(val)

df_val = pd.read_csv(os.path.join('data', 'mu_probe.csv'))

row_val = df_val['User Number'].values - 1
col_val = df_val['Movie Number'].values - 1
val_val = df_val['Rating'].values

n_samples_val = len(val_val)

n_users = 1 + np.max(row)
n_movies = 1 + np.max(col)
order = np.random.permutation(n_samples)

gc.collect()

print('Initializing model...')

batch = 100000
epochs = 20
i, j, r, se, pred, model = SVDpp(n_samples, n_users, n_movies, np.mean(val), mpu, lf=100, reg=0.02, learning_rate=5e-3)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)
print(sess.graph_def.ByteSize())

print('Training model...')

for e in range(epochs):
    sq_errs = []
    sq_errs_val = []
    start = clock()
    
    for prog, p in enumerate(range(int(n_samples // batch))):
        idx = order[np.arange(batch) + prog * batch]
        sess.run(model, feed_dict={i: row[idx], j: col[idx], r: val[idx]})
        
        c = sess.run(se, feed_dict={i: row[idx], j: col[idx], r: val[idx]})
        sq_errs.append(c)
        
    for prog, p in enumerate(range(int(n_samples_val // batch))):
        idx = np.arange(batch) + prog * batch
        c = sess.run(se, feed_dict={i: row_val[idx], j: col_val[idx], r: val_val[idx]})
        sq_errs_val.append(c)
    
    end = clock()
    train_rmse = np.sqrt(np.sum(sq_errs)/n_samples)
    val_rmse = np.sqrt(np.sum(sq_errs_val)/n_samples_val)
    t = end - start
    print('Epoch %d\t\tTrain RMSE = %.4f\tVal RMSE = %.4f\t\tTime = %.4f' % (e, train_rmse, val_rmse, t))
    

if submit:
    
    print('Saving submission...')
    df_qual = pd.read_csv(os.path.join('data', 'mu_qual.csv'))

    row_qual = df_qual['User Number'].values - 1
    col_qual = df_qual['Movie Number'].values - 1
    n_samples_qual = len(row_qual)
    
    predictions = []
    for prog, p in enumerate(range(1+ int(n_samples // batch))):
        l = prog * batch
        r = (prog + 1) * batch
        pr = sess.run(pred, feed_dict={i: row[l:r], j: col[l:r]})
        
        predictions += list(pr)
    
    save_submission(model_name, predictions, ordering)
