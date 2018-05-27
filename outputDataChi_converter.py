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
        fout.writelines(data[lines-1:]) 

# we know that we have to split the file into two different files at line
# 2749898, because these are how many data points we have for qual
def splitFile(fileName, label1, label2):
    with open(fileName, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(label1, 'w') as fout:
        fout.writelines(data[:2749899])
    with open(label2, 'w') as fout1:
        fout1.writelines(data[2749898:])


def deleteColumns(fileName):
    df = pd.read_csv(fileName, delim_whitespace=True, low_memory=False)
    # modify data fram to get rid of data we're not using
    del df[list(df.columns.values)[0]]
    del df[list(df.columns.values)[0]]

    print('first two columns removed, only have ratings left')
    
    df.to_csv(fileName, sep=' ', index=False, header=False)
    print('ratings saved in', fileName) 

        
def fileToPrediction_MU(fileName, label, label1, lines):
    newFileName = './submissions/' + label
    newFileName1 = './submissions/' + label1

    print('file', fileName, 'copied to', newFileName)
    copyfile(fileName, newFileName)
    removeFrontOfFile(newFileName, lines)
    print('first',str(lines), 'lines removed from Matrix Market format')
    
    splitFile(newFileName, newFileName, newFileName1)

    deleteColumns(newFileName)
    deleteColumns(newFileName1)
    #df = pd.read_csv(os.path.join('graphchi-cpp', fileName))
    
    
model = 'rbm'
fileOutLabelPred = 'mu_'+ model + '_graphchi' + strftime('%b%d%h%M%S') + '.pred'
fileOutLabelProbe = 'mu_' + model + '_graphchi' + strftime('%b%d%h%M%S') + '.probe'

#fileToPrediction_MU('./graphchi-cpp/mu_qual_val_mm.predict', fileOutLabelPred, fileOutLabelVal, 3)
fileToPrediction_MU('./graphchi-cpp/mu_qual_probe_mm.predict', fileOutLabelPred, fileOutLabelProbe, 3)

