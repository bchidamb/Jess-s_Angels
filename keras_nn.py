import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.utils import to_categorical
from utils import *

# WARNING: This file takes a very long time to run

model = 'keras_nn'
ordering = 'mu' # rows correspond to movie_ids; cols correspond to user_ids
submit = False # set to True to save a submission on qual


def embedding_model(n_users, n_movies):
    # Returns fresh Keras model for training on movie/user/rating data
    # X = user one-hot vector concatenated with movie one-hot vector
    # Y = integer rating (1-5)
    model = Sequential()
    model.add(Dense(10, input_dim=n_users + n_movies))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='SGD')
    return model
    
    
print('Loading data...')
df = pd.read_csv(os.path.join('data', 'mu_probe.csv')) # change to 'mu_train.csv'

row = df['User Number'].values - 1
col = df['Movie Number'].values - 1
val = df['Rating'].values

df_val = pd.read_csv(os.path.join('data', 'mu_val.csv'))

row_val = df_val['User Number'].values - 1
col_val = df_val['Movie Number'].values - 1
val_val = df_val['Rating'].values

n_users, n_movies = max(df['User Number']), max(df['Movie Number'])
n_examples = df.shape[0]
ind = np.random.permutation(n_examples)

def generate_examples(batch_size=100):
    
    while True:
        i = np.random.randint(n_examples // batch_size)        
        idx = ind[i * batch_size : (i + 1) * batch_size]
        user_vect = to_categorical(row[idx], n_users)
        movie_vect = to_categorical(col[idx], n_movies)
        x = np.hstack((user_vect, movie_vect))
        y = val[idx]
        
        yield (x, y)
        

print('Training model...')
batch_size = 100
cf_model = embedding_model(n_users, n_movies)
cf_model.fit_generator(
    generate_examples(batch_size), 
    steps_per_epoch=n_examples / batch_size, 
    epochs=1
)
