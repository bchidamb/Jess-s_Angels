import pandas as pd
import numpy as np
import os
from shutil import copyfile
from time import strftime

# This is so that we can use graphchi with our data, for our predicitions
# So this is so that we can feed in the prediction files that we generated using graphchi
# Matrix Market Exchange Formats
# https://math.nist.gov/MatrixMarket/formats.html
# https://people.sc.fsu.edu/~jburkardt/data/mm/mm.html



# delete from the front of the file 
# https://stackoverflow.com/questions/20364396/how-to-delete-the-first-line-of-a-text-file-using-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

        
def removeFrontOfFile(fileName, lines):
    with open(fileName, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(fileName, 'w') as fout:
        fout.writelines(data[lines:]) 
        
def fileToPrediction_MU(fileName, label, lines):
    newFileName = './submissions/' + label
    print('file', fileName, 'copied to', newFileName)
    copyfile(fileName, newFileName)
    removeFrontOfFile(newFileName, lines)
    print('first',str(lines), 'lines removed from Matrix Market format')
    
    #df = pd.read_csv(os.path.join('graphchi-cpp', fileName))
    df = pd.read_csv(newFileName, delim_whitespace=True, low_memory=False)
    # modify data fram to get rid of data we're not using
    del df[list(df.columns.values)[0]]
    del df[list(df.columns.values)[0]]

    print('first two columns removed, only have ratings left')
    
    df.to_csv(newFileName, sep=' ', index=False, header=False)
    print('ratings saved in', newFileName) 
    
    
fileOutLabel = 'mu_rbm' + strftime('%b%d%h%M%S') + '.pred'
fileToPrediction_MU('./graphchi-cpp/mu_qual_mm.predict', fileOutLabel, 3)


