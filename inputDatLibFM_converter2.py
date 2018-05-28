import pandas as pd
import numpy as np
import os
import gc

# The whole goal of this file is to now convert our data into the format that is used for 
# ratings.dat in the movielens dataset which is:
# UserID::MovieID::Rating::Timestamp
# once we do this, we can use a pearl script built into libFM to convert these ratings
# and then make predictions
# http://files.grouplens.org/datasets/movielens/ml-10m-README.html
# https://github.com/srendle/libfm
# http://www.libfm.org/libfm-1.42.manual.pdf

# wrote this function knowing what was inside of our files with our columns
def fileToLibfm_MU(fileName, label):
    # MU data
    print('Loading data', label,'mu...')
    df = pd.read_csv(os.path.join('data', fileName))
    # modify data fram to get rid of data we're not using
    #del df['Unnamed: 0']
    #del df['Unnamed: 0.1']
    #del df['bin'] # from our bin stuff
    #del df['Date Number']
    df = df.astype('int32')

    newFileName = label + '_libFM'
    newFileLocation = "libfm/" + newFileName

    print('Making new file', newFileName) 

    # make a new file the old fashion way
    f = open(newFileLocation, "w+")
    stringToAdd = ''
    count = 0
    for index, row in df.iterrows():
        stringToAdd += str(row['User Number']) + "::" + str(row['Movie Number']) + "::" + str(row['Rating']) + '::' + str(row['Date Number']) #+ '\n' 
        count += 1
        if count is 50000:
            f.write(stringToAdd)
            stringToAdd = ''
            count = 0
        else:
            stringToAdd += '\n'

    if stringToAdd is not '':
        f.write(stringToAdd[:-1]) # need to take off last '\n'

    f.close()
    gc.collect()
            
    print('Finished reading in data')
    print('file processing done for', label, 'new file created', newFileLocation, '\n')
    

#fileToLibfm_MU('real_mu_qual_probe.csv', 'real_mu_qual_probe')
#fileToLibfm_MU('mu_val.csv', 'mu_val')
fileToLibfm_MU('real_mu_train.csv', 'real_mu_train')
#fileToLibfm_MU('mu_probe.csv', 'mu_probe')
#fileToLibfm_MU('mu_qual.csv', 'mu_qual')
#fileToLibfm_MU('mu_qual_val.csv', 'mu_qual_val')


