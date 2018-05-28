import numpy as np
import pandas as pd
import os
from time import strftime, clock


def RMSE(true, pred):
    # computes Root Mean-squared-error given true ratings and predictions
    return np.sqrt(np.sum((true - pred)**2) / len(true))
    
    
def f_time(f, *args, **kwargs):
    '''
    Runs f on some input and prints the time elapsed until an output is returned
    
    Arguments:
        f - the function to run
        *args - the positional arguments to f
        *kwargs - the keyword arguments to f
    
    Returns:
        Output of running f on input
    '''
    start = clock()
    output = f(*args, **kwargs)
    print('Function runtime: %.2f s' % (clock() - start))
    return output
    
    
def save_submission(model_name, pred, ordering='mu'):
    '''
    Saves submission on qual set given predictions
    File is saved as 'submissions/*.pred'
    
    Arguments:
        model_name - a string identifying the model used to predict
        pred - a list of numbers giving the rating predictions on qual
        ordering - 'mu' for mu_qual.csv; 'um' for um_qual.csv
    '''
    filename = '_'.join([ordering, model_name, strftime('%b%d%H%M%S')]) + '.pred'
    f = open(os.path.join('submissions', filename), 'w')
    
    for p in pred:
        f.write('%.3f\n' % p)
        
    f.close()
    
    
def um_to_mu(filename, dataset=os.path.join('data', 'um_qual.csv')):
    '''
    Takes a submission file saved in user-movie ordering and swaps it to 
    movie-user ordering
    
    Arguments:
        filename - the path of the submission file for um_qual
        dataset - the path of the dataset whose predictions are converted
    
    Returns:
        A list of numbers holding the predictions in movie-user order
    '''
    qual = pd.read_csv(dataset)
    um_pred = np.loadtxt(filename, delimiter='\n')
    
    qual['pred'] = pd.Series(um_pred, index=qual.index)
    
    mu_qual = qual.sort_values(['Movie Number', 'User Number'])['pred'].values
    
    return mu_qual
    
    