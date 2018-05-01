import numpy as np
import pandas as pd
from utils import *
import os

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing 


model = 'sklearn_RBM'
ordering = 'mu' # rows correspond to movie_ids; cols correspond to user_ids
submit = True # set to True to save a submission on qual
save_model = False # set to True to save model parameters for future predictions

print('Loading data...')
df = pd.read_csv(os.path.join('data', 'mu_train.csv'))
# modify dataframe to reduce memory
del df['Unnamed: 0']
del df['Date Number']
df = df.astype('int32')


#print('df', type(df)) # user number, movie number, rating
#print(df)

#print(datasets.load_digits().iloc[0])

df_npArray = df.as_matrix

Y = df['Rating'].values
#print('Y:', Y)

del df['Rating']

df['Movie Number'] = (df['Movie Number'] - df['Movie Number'].min()) / (df['Movie Number'].max() - df['Movie Number'].min())

df['User Number'] = (df['User Number'] - df['User Number'].min()) / (df['User Number'].max() - df['User Number'].min())

#normal_df = (df - df.min()) / (df.max() - df.min())


X = df.values

print('type X', type(X), len(X), len(X[0]), 'type Y', type(Y), len(Y))
#print(X)

#X = (X - np.amin(X, 0)) / (np.amax(X, 0) + 0.0001)  # 0-1 scaling for X values

# split into training and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2,
                                                    random_state=0)

print(len(X_train), len(X_train[0]))

# models we'll use
logReg = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)

classifier = Pipeline(steps=[('rbm',rbm),('logReg',logReg)])

#rbm.learning_rate = 0.06
rbm.n_iter = 10
rbm.n_componets = 100
rbm.batch_size = 1
rbm.verbose = 1

logReg.C = 6000.0

print('Training model...')
# Train RBM-LogReg Pipeline

classifier.fit(X_train, Y_train)

print('Training completed!')

Y_predict = classifier.predict(X_test)


error = mean_squared_error(Y_test, Y_predict)
print('Mean squared error on test:', error) 




