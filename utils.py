import numpy as np
import os
from time import strftime


def RMSE(true, pred):
    return np.sqrt(np.sum((true - pred)**2) / len(true))
    
    
def save_submission(pred):
    # Saves submission on qual set given predictions
    # File is saved as 'submissions/*.pred'
    filename = '_'.join([ordering, model, strftime('%b%d%H%M%S')]) + '.pred'
    f = open(os.path.join('submissions', filename), 'w')
    
    for p in pred:
        f.write('%.3f\n' % p)
        
    f.close()