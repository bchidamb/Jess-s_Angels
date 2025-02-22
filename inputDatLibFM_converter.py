import pandas as pd
import numpy as np
import os

# The whole goal of this file is to now convert our data into the format that is used for 
# ratings.dat in the movielens dataset which is:
# UserID::MovieID::Rating::Timestamp
# once we do this, we can use a pearl script built into libFM to convert these ratings
# and then make predictions
# http://files.grouplens.org/datasets/movielens/ml-10m-README.html
# https://github.com/srendle/libfm
# http://www.libfm.org/libfm-1.42.manual.pdf

# prepend to the file 
# https://www.quora.com/How-can-I-write-text-in-the-first-line-of-an-existing-file-using-Python
#def prependEntries(fileName, string1, string2):
#    with open(fileName, 'r+') as f:
#        file_data = f.read()
#        f.seek(0,0)
#        f.write(string1.rstrip('\r\n') + '\n' + string2.rstrip('\r\n') + '\n' + file_data)

# wrote this function knowing what was inside of our files with our columns
def fileToMatrixMarket_MU(fileName, label):
    # MU data
    print('Loading data', label,'mu...')
    df = pd.read_csv(os.path.join('data', fileName))
    # modify data fram to get rid of data we're not using
    del df['Unnamed: 0']
    #del df['Date Number']
    def df['bin'] # from our bin stuff
    df = df.astype('int32')

    # move the ratings to the last column, and date number to the third column
    cols = df.columns.tolist() # user, movie, date, rating -> user, movie, rating, date
    cols = cols[:2] + list(cols[3]) + list(cols[2]) # rearrange columsn
    df = df[cols]
    
    # assume that our first read in file has the number of users and movies
    #if maxUsers == -1:
    #    maxUsers = df['User Number'].max()
    #    maxMovies = df['Movie Number'].max()

    #numRatings = df.shape[0]
    
    #print('maxUsers:', maxUsers, 'maxMovies:', maxMovies, 'numRatings', numRatings)

    # append 0: to the 2nd column, 1: to the 3rd, because data fomat is:
    # y 0:x1 1:x2 ... 

    newFileName = label + '_libFM'
    newFileLocation = "libfm/" + newFileName

    print('Making new file', newFileName) 
    df.to_csv(newFileLocation, sep='::', index=False, header=False)
    print('Finished reading in data')
    
    # need this for the first row for the Matrix Market Exchange Format
    #rowsColsEntries = str(maxUsers) + ' ' + str(maxMovies) + ' ' + str(numRatings)
    #print('num Users, num Movies, num ratings', rowsColsEntries)
    #fileHeader = '%%MatrixMarket matrix coordinate real general'
    #prependEntries(newFileLocation, fileHeader, rowsColsEntries)
    
    print('file processing done for', label, 'new file created', newFileLocation, '\n')
    


fileToMatrixMarket_MU('mu_train.csv', 'mu_train')
fileToMatrixMarket_MU('mu_val.csv', 'mu_val')
fileToMatrixMarket_MU('mu_probe.csv', 'mu_probe')
fileToMatrixMarket_MU('mu_qual.csv', 'mu_qual')
fileToMatrixMarket_MU('mu_qual_val.csv', 'mu_qual_val')
fileToMatrixMarket_MU('mu_qual_probe.csv', 'mu_qual_probe')

