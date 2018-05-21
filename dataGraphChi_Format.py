import pandas as pd
import numpy as np
import os

# So this is so that we can feed in the .csv files that we already had into the 
# Matrix Market Exchange Formats
# https://math.nist.gov/MatrixMarket/formats.html
# https://people.sc.fsu.edu/~jburkardt/data/mm/mm.html
# Need to make my code have functions later, hopefully this works lol


# MU data
print('Loading data train mu...')
df = pd.read_csv(os.path.join('data', 'mu_train.csv'))
# modify data fram to get rid of data we're not using
del df['Unnamed: 0']
del df['Date Number']
df = df.astype('int32')

maxUsers = df['User Number'].max()
maxMovies = df['Movie Number'].max()
numRatings = df.shape[0]

print('maxUsers:', maxUsers, 'maxMovies:', maxMovies, 'numRatings', numRatings)

print('Making new file mu_train_mm')
df.to_csv("graphchi-cpp/mu_train_mm", sep=' ', index=False, header=False)
print('Finished reading in data, need to prepend info')

# need this for the first row for the Matrix Market Exchange Format

rowsColsEntries = str(maxUsers) + ' ' + str(maxMovies) + ' ' + str(numRatings)
print('num Users, num Movies, num ratings', rowsColsEntries)
fileHeader = '%%MatrixMarket matrix coordinate real general'

# prepend to the file 
# https://www.quora.com/How-can-I-write-text-in-the-first-line-of-an-existing-file-using-Python
with open('graphchi-cpp/mu_train_mm', 'r+') as f:
    file_data = f.read()
    f.seek(0,0)
    f.write(fileHeader.rstrip('\r\n') + '\n' + rowsColsEntries.rstrip('\r\n') + '\n' + file_data)
    
print('file processing done for train, now val')
#==============================================================================
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# MONEY MONEY DOLLA$ Y'ALL
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

print('Loading data validation mu...')
df = pd.read_csv(os.path.join('data', 'mu_val.csv'))
# modify data fram to get rid of data we're not using
del df['Unnamed: 0']
del df['Date Number']
df = df.astype('int32')

maxUsers = df['User Number'].max()
maxMovies = df['Movie Number'].max()
numRatings = df.shape[0]

print('maxUsers:', maxUsers, 'maxMovies:', maxMovies, 'numRatings', numRatings)

print('Making new file mu_val_mm')
df.to_csv("graphchi-cpp/mu_val_mm", sep=' ', index=False, header=False)
print('Finished reading in data, need to prepend info')

# need this for the first row for the Matrix Market Exchange Format

rowsColsEntries = str(maxUsers) + ' ' + str(maxMovies) + ' ' + str(numRatings)
print('num Users, num Movies, num ratings', rowsColsEntries)
fileHeader = '%%MatrixMarket matrix coordinate real general'

# prepend to the file 
# https://www.quora.com/How-can-I-write-text-in-the-first-line-of-an-existing-file-using-Python
with open('graphchi-cpp/mu_val_mm', 'r+') as f:
    file_data = f.read()
    f.seek(0,0)
    f.write(fileHeader.rstrip('\r\n') + '\n' + rowsColsEntries.rstrip('\r\n') + '\n' + file_data)

    
print('file processing done for val, we done!')

