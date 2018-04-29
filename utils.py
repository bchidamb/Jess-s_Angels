import numpy as np
import os
from time import strftime


def RMSE(true, pred):
    # computes Root Mean-squared-error given true ratings and predictions
    return np.sqrt(np.sum((true - pred)**2) / len(true))
    
    
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