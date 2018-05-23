import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Activation, Dense
from utils import *
import os
import gc

# Input the filenames of the prediction files for all modles
probe_names = ["mu_surprise_svd_Apr30041151.pred"]
qual_names = []
num_models = len(probe_names)

ordering = 'mu' # rows correspond to movie_ids; cols correspond to user_ids
submit = True# set to True to save a submission on qual
save_model = False # set to True to save model parameters for future predictions


# computes Root Mean-squared-error given true ratings and predictions
def RMSE(true, pred):
    return np.sqrt(np.sum((true - pred)**2) / len(true))


# Load the probe and qual set predictions to compare against
print('Loading probe set...')
probeSet = pd.read_csv(os.path.join('data', 'mu_probe.csv'))
# modify dataframe to reduce memory
del probeSet['Unnamed: 0']
del probeSet['Date Number']
probeSet = np.array(probe.astype('int32'))

print('Loading qual set...')
qualSet = pd.read_csv(os.path.join('data', 'mu_qual.csv'))
# modify dataframe to reduce memory
del qualSet['Unnamed: 0']
del qualSet['Date Number']
qualSet = np.array(qual.astype('int32'))



# Obtain the predictions
probe_list = []
for probe_name in probe_names:
    probe = pd.read_csv(os.path.join('submissions', probe_name))
    probe = probe.astype('int32')
    print("RMSE for", probe_name, ":", RMSE(probeSet, probe))
    probe_list.append(np.array(probe))
probe_list = np.array(probe_list).T

qual_list = []
for qual_name in qual_names:
    qual = pd.read_csv(os.path.join('submissions', qual_name))
    qual = qual.astype('int32')
    print("RMSE for", qual_name, ":", RMSE(qualSet, qual))
    qual_list.append(np.array(qual))
qual_list = np.array(qual_list).T


# Feed into neural network
nnmodel = Sequential()
nnmodel.add(Dense(1, input_dim=num_models))
nnmodel.fit(probe_list, probeSet)

blend_pred = nnmodel.predict(qual_list)
blend_RMSE = RMSE(qualSet, blend_pred)
print("RMSE for blended model:", blend_RMSE)


if save_model:

    print('Saving model...')
    dump.dump(os.path.join('models', 'blend'), nnmodel)

# Profit 
if submit:
    save_submission("blend", blend_pred, ordering)
