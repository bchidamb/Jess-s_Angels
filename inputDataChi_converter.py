import pandas as pd
import numpy as np
import os

# This is so that we can use graphchi with our data, for our predicitions
# So this is so that we can feed in the .csv files that we already had into the 
# Matrix Market Exchange Formats
# https://math.nist.gov/MatrixMarket/formats.html
# https://people.sc.fsu.edu/~jburkardt/data/mm/mm.html

# prepend to the file 
# https://www.quora.com/How-can-I-write-text-in-the-first-line-of-an-existing-file-using-Python
def prependEntries(fileName, string1, string2):
    with open(fileName, 'r+') as f:
        file_data = f.read()
        f.seek(0,0)
        f.write(string1.rstrip('\r\n') + '\n' + string2.rstrip('\r\n') + '\n' + file_data)

# wrote this function knowing what was inside of our files with our columns
def fileToMatrixMarket_MU(fileName, label, maxUsers, maxMovies):
    # MU data
    print('Loading data', label,'mu...')
    df = pd.read_csv(os.path.join('data', fileName))
    # modify data fram to get rid of data we're not using
    del df['Unnamed: 0']
    del df['Date Number']
    df = df.astype('int32')
    
    # assume that our first read in file has the number of users and movies
    if maxUsers == -1:
        maxUsers = df['User Number'].max()
        maxMovies = df['Movie Number'].max()

    numRatings = df.shape[0]
    
    print('maxUsers:', maxUsers, 'maxMovies:', maxMovies, 'numRatings', numRatings)
    newFileName = label + '_mm'
    newFileLocation = "graphchi-cpp/" + newFileName

    print('Making new file', newFileName) 
    df.to_csv(newFileLocation, sep=' ', index=False, header=False)
    print('Finished reading in data, need to prepend info')
    
    # need this for the first row for the Matrix Market Exchange Format
    rowsColsEntries = str(maxUsers) + ' ' + str(maxMovies) + ' ' + str(numRatings)
    print('num Users, num Movies, num ratings', rowsColsEntries)
    fileHeader = '%%MatrixMarket matrix coordinate real general'
    
    prependEntries(newFileLocation, fileHeader, rowsColsEntries)
    
    print('file processing done for', label, 'new file created', newFileLocation, '\n')
    
    return(maxUsers, maxMovies)


maxUsers = -1
maxMovies = -1

maxUsers, maxMovies = fileToMatrixMarket_MU('mu_train.csv', 'mu_train', maxUsers, maxMovies)
fileToMatrixMarket_MU('mu_val.csv', 'mu_val', maxUsers, maxMovies)
fileToMatrixMarket_MU('mu_probe.csv', 'mu_probe', maxUsers, maxMovies)
fileToMatrixMarket_MU('mu_qual.csv', 'mu_qual', maxUsers, maxMovies)
fileToMatrixMarket_MU('mu_qual_val.csv', 'mu_qual_val', maxUsers, maxMovies)
fileToMatrixMarket_MU('mu_qual_probe.csv', 'mu_qual_probe', maxUsers, maxMovies)


