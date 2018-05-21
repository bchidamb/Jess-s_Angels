import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy, dump
from surprise import AlgoBase, NMF, SVDpp, CoClustering
from keras.models import Sequential
from keras.layers import Activation, Dense
from random import randint
from utils import *
import os
import gc

# WARNING: This file takes a long time to run

model_name = 'surprise_SVD'
ordering = 'mu' # rows correspond to movie_ids; cols correspond to user_ids
submit = True# set to True to save a submission on qual
save_model = False # set to True to save model parameters for future predictions


print('Loading data...')
df = pd.read_csv(os.path.join('data', 'mu_train.csv'))
# modify dataframe to reduce memory
del df['Unnamed: 0']
del df['Date Number']
df = df.astype('int32')

df_val = pd.read_csv(os.path.join('data', 'mu_val.csv'))


reader = Reader(rating_scale=(1, 5))
train_raw = Dataset.load_from_df(df[['User Number', 'Movie Number', 'Rating']], reader)
subset_length = 1000000
# Take a subset if you want
# train_full = train_raw
#train_full = train_raw[:subset_length]

train = train_raw.build_full_trainset()



print('Solving SVD...')
model = SVD(n_epochs = 20, verbose=True)
model.fit(train)
gc.collect()

# Model for NMF
#print("Solving NMF Matrix Factorization")
#nmf_model = NMF(n_factors=5, n_epochs=20, verbose=True)
#nmf_model.fit(train)
#gc.collect()

# Model for CoCluster
#print("Solving CoCluster")
#cluster_model = CoCluster(n_cltr_u=7, n_cltr_i =5)
#cluster_model.fit(train)
#gc.collect()


def RMSE(true, pred):
    # computes Root Mean-squared-error given true ratings and predictions
    return np.sqrt(np.sum((true - pred)**2) / len(true))


#'''
train_pred = model.test(train.build_testset())
val_raw = Dataset.load_from_df(df_val[['User Number', 'Movie Number', 'Rating']], reader)
val = val_raw.build_full_trainset()
valset = val.build_testset()
svd_pred = model.test(valset)

print('Train RMSE:', accuracy.rmse(train_pred))
print('SVD VAl RMSE:', accuracy.rmse(svd_pred))
#
# nmf_pred = nmf_model.test(valset)
# print('NMF VAl RMSE:', accuracy.rmse(nmf_pred))
#
# cluster_pred = cluster_model.test(valset)
# print('Cluster VAl RMSE:', accuracy.rmse(cluster_pred))
#
# # blend
#
# #real_svd = np.zeros(len(svd_pred))
# num_models = 3
# blend_pred = np.zeros(len(svd_pred), num_models)
#
# for j, pred in enumerate(svd_pred):
#     blend_pred[j][0] = pred.est
#
# #real_nmf = np.zeros(len(nmf_pred))
# for j, pred in enumerate(nmf_pred):
#     blend_pred[j][1] = pred.est
#
# #real_cluster = np.zeros(len(cluster_pred))
# for j, pred in enumerate(cluster_pred):
#     blend_pred[j][2] = pred.est
#
# #blend_pred = blend_pred / 3
#
# # Use keras neural net
#
# nnmodel = Sequential()
# nnmodel.add(Dense(1, input_dim=num_models))
# nnmodel.fit(blend_pred, valset)
#

#real_blend_pred = nnmodel.predict(blend_pred)
#
# blend_rmse = RMSE(blend_pred, valset)
#
# save_submission(model_name, real_blend_pred, ordering)

#'''

if save_model:

    print('Saving model...')
    dump.dump(os.path.join('models', 'surprise_model'), model)

if submit:

    print('Saving submission...')
    df_qual = pd.read_csv(os.path.join('data', 'mu_qual.csv'))

    pred = []
    for _, row in df_qual.iterrows():
        r_est = model.predict(row['User Number'], row['Movie Number']).est
        pred.append(r_est)

    save_submission(model_name + "_qual", pred, ordering)

    # Now save the probe predictions as well
    df_qual = pd.read_csv(os.path.join('data', 'mu_probe.csv'))

    pred = []
    for _, row in df_qual.iterrows():
        r_est = model.predict(row['User Number'], row['Movie Number']).est
        pred.append(r_est)

    save_submission(model_name + "_probe", pred, ordering)
