import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import Concatenate, LSTM, Input, concatenate
from keras.optimizers import Adagrad
from utils import *
import os
import gc

# Assuming everything is in Movie User order


# Input the filenames of the prediction files for all modles
probe_names = ["submissions/mu_svd_probe_May23195204.pred",
                "pmf/predictionsl005k65t10"]
qual_names = ["submissions/mu_svd_qual_May23195205.pred",
                "pmf/predictionsl005k65t10qual"]
num_models = len(probe_names)

probe_length = 1374739
qual_length = 2749898

method = 1  #1: single layer, 2: 20-40 layers
n_hidden = 30
ordering = 'mu' # rows correspond to movie_ids; cols correspond to user_ids
submit = True # set to True to save a submission on qual
save_model = False # set to True to save model parameters for future predictions


# computes Root Mean-squared-error given true ratings and predictions
def RMSE(true, pred):
    return np.sqrt(np.sum((true - pred)**2) / len(true))


# Load the probe and qual set predictions to compare against
print('Loading probe set...')
probeSet = pd.read_csv(os.path.join('data', 'mu_probe.csv'))
# modify dataframe to reduce memory
#del probeSet['Unnamed: 0']
#del probeSet['Date Number']
probeSet = probeSet[["Rating"]]
probeSet = np.array(probeSet.astype('double'))

#print('Loading qual set...')
#qualSet = pd.read_csv(os.path.join('data', 'mu_qual.csv'))
# modify dataframe to reduce memory
#del qualSet['Unnamed: 0']
#del qualSet['Date Number']
#qualSet = qualSet[["Rating"]]
#qualSet = np.array(qualSet.astype('double'))



# Obtain the predictions
probe_list = []
for probe_name in probe_names:
    #probe = pd.read_csv(os.path.join('submissions', probe_name))
    probe = pd.read_csv(probe_name, header=None)
    probe = probe.astype('double')
    print("RMSE for", probe_name, ":", RMSE(probeSet, probe))
    probe_list.append(np.array(probe))
probe_list = np.array(probe_list).T
probe_list = probe_list[0:][0]

qual_list = []
for qual_name in qual_names:
    #qual = pd.read_csv(os.path.join('submissions', qual_name))
    qual = pd.read_csv(qual_name, header=None)
    qual = qual.astype('double')
    #print("RMSE for", qual_name, ":", RMSE(qualSet, qual))
    qual_list.append(np.array(qual))
qual_list = np.array(qual_list).T
qual_list = qual_list[0:][0]

print("probe train dim:", probe_list.ndim)
print("probe test dim:", probeSet.ndim)

print("qual train dim:", qual_list.ndim)
print("qual test dim:", qualSet.ndim)
#print(qual_list[0])


# Use a hidden factor of 20-40 weights
# and Relu activation
if method == 2:
    nnmodel = Sequential()
    nnmodel.add(Dense(n_hidden, input_dim=num_models, activation="relu"))
    nnmodel.add(Dense(1, input_dim=num_models, activation="sigmoid"))

elif method == 1:
    # Feed into neural network
    nnmodel = Sequential()
    nnmodel.add(Dense(1, input_dim=num_models))

nnmodel.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
nnmodel.fit(probe_list, probeSet, epochs = 5, batch_size=128)

blend_pred = nnmodel.predict(qual_list)
#blend_RMSE = RMSE(qualSet, blend_pred)
#print("RMSE for blended model:", blend_RMSE)

if save_model:

    print('Saving model...')
    dump.dump(os.path.join('models', 'blend'), nnmodel)

# Profit
if submit:
    save_submission("blend1", blend_pred, ordering)
