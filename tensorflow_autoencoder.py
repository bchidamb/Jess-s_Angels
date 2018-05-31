import tensorflow as tf
from tensorflow_utils import *
import numpy as np
import pandas as pd
import os
import sys
from time import clock
from data_processing import *
from utils import *


# useful links:
# http://users.cecs.anu.edu.au/~u5098633/papers/www15.pdf
# https://github.com/NVIDIA/DeepRecommender


def I_AutoRec(n_users, k, lr, reg, dpt):
    
    input = tf.placeholder(tf.float32, shape=[None, n_users])
    mask = tf.placeholder(tf.float32, shape=[None, n_users])
    
    inits = tf.truncated_normal_initializer(0, 0.05)
    regs = tf.contrib.layers.l2_regularizer(scale=reg)
    
    input_prc = tf.multiply(input, mask) # tf.layers.dropout(input * mask, rate=dpt)
    
    layer_args = {'kernel_initializer': inits, 
                    'bias_initializer': inits, 
                    'kernel_regularizer': regs}
    
    hidden1 = tf.layers.dense(input_prc, k, activation=tf.sigmoid, **layer_args)
    # hidden2 = tf.layers.dense(hidden1, k, activation=tf.sigmoid, **layer_args)
    predictions = tf.layers.dense(hidden1, n_users, activation=tf.sigmoid, **layer_args)
    
    se = tf.reduce_sum(tf.square(tf.multiply(predictions - input, mask))) / tf.reduce_sum(mask)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = se + reg_losses
    
    model = tf.train.AdamOptimizer(lr).minimize(loss)
    
    return input, mask, predictions, model

    
'''
def DeepAutoencoder(n_movies):

    input = tf.placeholder(tf.float32, shape=[n_movies, None])
    
    initializer = tf.
    
    W1 = tf.get_variable("W1", [], inits = , regularizer = )
    b1 = tf
    X1 = b1
    
    return 
'''


print('Loading data...')

df = pd.read_csv(os.path.join('data', 'mu_train.csv'))

row = df['User Number'].values - 1
col = df['Movie Number'].values - 1
val = df['Rating'].values

n_samples = len(val)

n_users = 1 + np.max(row)
n_movies = 1 + np.max(col)

L = col.searchsorted(np.arange(n_movies), side='left')
R = col.searchsorted(np.arange(n_movies), side='right')

df_val = pd.read_csv(os.path.join('data', 'mu_probe.csv'))

row_val = df_val['User Number'].values - 1
col_val = df_val['Movie Number'].values - 1
val_val = df_val['Rating'].values

n_samples_val = len(val_val)

L_val = col_val.searchsorted(np.arange(n_movies), side='left')
R_val = col_val.searchsorted(np.arange(n_movies), side='right')

print('Training model...')

input, mask, predictions, model = I_AutoRec(n_users, 50, 0.01, 0.001, 0.0)
epochs = 20
batch = 20

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print('Model size: %d bytes' % sess.graph_def.ByteSize())

for e in range(epochs):
    start = clock()
    
    order = np.random.permutation(n_movies)
    
    pred_train = np.zeros(n_samples)
    pred_val = np.zeros(n_samples_val)
    
    for b in range(int(1 + n_movies // batch)):
        
        example, binary = np.zeros((batch, n_users)), np.zeros((batch, n_users))
        
        for i, o in enumerate(range(b * batch, min((b + 1) * batch, n_movies))):
            m = order[o]
            # transform ratings to be between 0 and 1 (inclusive)
            example[i, row[L[m]:R[m]]] = (val[L[m]:R[m]] - 1.0) / 4.0
            binary[i, row[L[m]:R[m]]] = 1
        
        m_pred, _ = sess.run([predictions, model], feed_dict={input: example, mask: binary})
        # undo transform to obtain true ratings
        m_pred = 1.0 + 4.0 * m_pred
        
        for i, o in enumerate(range(b * batch, min((b + 1) * batch, n_movies))):
            m = order[o]
            pred_train[L[m]:R[m]] = m_pred[i, row[L[m]:R[m]]]
            pred_val[L_val[m]:R_val[m]] = m_pred[i, row_val[L_val[m]:R_val[m]]]
        
    end = clock()
    train_rmse = RMSE(pred_train, val)
    val_rmse = RMSE(pred_val, val_val)
    t = end - start
    print('Epoch %d\t\tTrain RMSE = %.4f\tVal RMSE = %.4f\t\tTime = %.4f' % (e, train_rmse, val_rmse, t))