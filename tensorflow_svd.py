import tensorflow as tf
from tensorflow_utils import *
import numpy as np
import pandas as pd
import os
from time import clock
from utils import *

submit = False
model_name = 'tensorflow_svd'
ordering = 'mu'


# useful links:
# https://github.com/aymericdamien/TensorFlow-Examples
# https://github.com/songgc/TF-recomm
# http://surprise.readthedocs.io/en/stable/matrix_factorization.html


def SVD(n_samples, n_u, n_m, mean, lf=100, reg=0.02, learning_rate=0.005):

    i = tf.placeholder(tf.int32, shape=[None])
    j = tf.placeholder(tf.int32, shape=[None])
    r = tf.placeholder(tf.float32, shape=[None])
    
    batch = tf.shape(r)[0]
    
    inits = tf.random_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32)
    mu = tf.constant([mean])
    b_u = tf.get_variable('user_bias', shape=[n_u], initializer=inits)
    b_m = tf.get_variable('movie_bias', shape=[n_m], initializer=inits)
    emb_u = tf.get_variable('user_embedding', shape=[n_u, lf], initializer=inits)
    emb_m = tf.get_variable('movie_embedding', shape=[n_m, lf], initializer=inits)
    
    to_sum = []
    slice = tf.nn.embedding_lookup
    to_sum.append(tf.tile(mu, [batch]))
    to_sum.append(slice(b_u, i))
    to_sum.append(slice(b_m, j))
    to_sum.append(tf.reduce_sum(tf.multiply(slice(emb_u, i), slice(emb_m, j)), 1))
    
    # optimize for sum(squared error) with l2 regularization
    l2 = tf.nn.l2_loss(slice(emb_u, i)) \
        + tf.nn.l2_loss(slice(emb_m, j))\
        + tf.nn.l2_loss(slice(b_u, i))  \
        + tf.nn.l2_loss(slice(b_u, j))
    r_pred = tf.add_n(to_sum)
    train_loss = tf.reduce_sum(tf.pow(r_pred-r,2)) + reg * l2
    se = tf.reduce_sum(tf.pow(r_pred-r,2))
    model = tf.train.GradientDescentOptimizer(learning_rate).minimize(train_loss)
    
    return i, j, r, se, r_pred, model
    

print('Loading data...')

df = pd.read_csv(os.path.join('data', 'mu_train.csv'))

row = df['User Number'].values - 1
col = df['Movie Number'].values - 1
val = df['Rating'].values

n_samples = len(val)

df_val = pd.read_csv(os.path.join('data', 'mu_probe.csv'))

row_val = df_val['User Number'].values - 1
col_val = df_val['Movie Number'].values - 1
val_val = df_val['Rating'].values

n_samples_val = len(val_val)

n_users = 1 + np.max(row)
n_movies = 1 + np.max(col)
order = np.random.permutation(n_samples)


print('Training model...')

batch = 10000
epochs = 20
i, j, r, se, pred, model = SVD(n_samples, n_users, n_movies, np.mean(val), lf=200, reg=0.02, learning_rate=5e-3)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

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

    for dataset in ('qual', 'probe'):
    
        print('Saving submission...')
        df_qual = pd.read_csv(os.path.join('data', 'mu_' + dataset + '.csv'))

        row_qual = df_qual['User Number'].values - 1
        col_qual = df_qual['Movie Number'].values - 1
        
        n_samples_qual = len(row_qual)
        
        predictions = []
        for prog, p in enumerate(range(1+ int(n_samples_qual // batch))):
            li = prog * batch
            ri = min((prog + 1) * batch, n_samples_qual)
            pr = sess.run(pred, feed_dict={i: row_qual[li:ri], j: col_qual[li:ri], r: np.zeros(ri - li)})
            
            predictions += list(pr)
        
        save_submission(model_name + '_' + dataset, predictions, ordering)