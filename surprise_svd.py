import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy, dump
from utils import *
import os

# WARNING: This file takes a long time to run

model = 'surprise_svd'
ordering = 'mu' # rows correspond to movie_ids; cols correspond to user_ids
submit = False # set to True to save a submission on qual
save_model = False # set to True to save model parameters for future predictions


print('Loading data...')
df = pd.read_csv(os.path.join('data', 'mu_train.csv'))
# modify dataframe to reduce memory
del df['Unnamed: 0']
del df['Date Number']
df = df.astype('int32')

df_val = pd.read_csv(os.path.join('data', 'mu_val.csv'))


print('Solving SVD...')
reader = Reader(rating_scale=(1, 5))
model = SVD(n_epochs = 20, verbose=True)

train_raw = Dataset.load_from_df(df[['User Number', 'Movie Number', 'Rating']], reader)
train = train_raw.build_full_trainset()

model.fit(train)

train_pred = model.test(train.build_testset())
val_raw = Dataset.load_from_df(df_val[['User Number', 'Movie Number', 'Rating']], reader)
val = val_raw.build_full_trainset()
val_pred = model.test(val.build_testset())

print('Train RMSE:', accuracy.rmse(train_pred))
print('Val RMSE:', accuracy.rmse(val_pred))

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
        
    save_submission(model, pred, ordering)