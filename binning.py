import pandas as pd
import numpy as np
import os
import gc

# Puts all ratings into one of 30 bins
num_bins = 30

# Min and max date as specified in the README
minBin = 1.0
maxBin = 2243.0
binSize = float((maxBin - minBin)/num_bins)

def binDataframe(df):

    # Create a new column called bin
    df["bin"] = pd.Series([-1]*len(df), index=df.index)

    # Divide the times into even bins
    #minBin = min(df["Date Number"])
    #maxBin = max(df["Date Number"])

    for i in range(0, num_bins):
        lb = minBin + i*binSize
        ub = lb + binSize

        df.loc[(df["Date Number"] >= lb) & (df["Date Number"] <= ub), ["bin"]] = i
        #print(df.loc[(df["Date Number"] >= lb) & (df["Date Number"] <= ub)]["bin"])

    return df

# Perform this for all datasets

# UM Data sets
print('Loading um probe set...')
um_probe = pd.read_csv(os.path.join('data', 'um_probe.csv'))
um_probe = binDataframe(um_probe)
um_probe.to_csv("data/um_probe.csv")
gc.collect()

print('Loading um val set...')
um_val = pd.read_csv(os.path.join('data', 'um_val.csv'))
um_val = binDataframe(um_val)
um_val.to_csv("data/um_val.csv")
gc.collect()

print('Loading um hidden set...')
um_hidden = pd.read_csv(os.path.join('data', 'um_hidden.csv'))
um_hidden = binDataframe(um_hidden)
um_hidden.to_csv("data/um_hidden.csv")
gc.collect()

print('Loading um qual set...')
um_qual = pd.read_csv(os.path.join('data', 'um_qual.csv'))
um_qual = binDataframe(um_qual)
um_qual.to_csv("data/um_qual.csv")
gc.collect()

print('Loading um train set...')
um_train = pd.read_csv(os.path.join('data', 'um_train.csv'))
um_train = binDataframe(um_train)
um_train.to_csv("data/um_train.csv")
gc.collect()

# MU Data Sets
print('Loading mu probe set...')
mu_probe = pd.read_csv(os.path.join('data', 'mu_probe.csv'))
mu_probe = binDataframe(mu_probe)
mu_probe.to_csv("data/mu_probe.csv")
gc.collect()

print('Loading mu val set...')
mu_val = pd.read_csv(os.path.join('data', 'mu_val.csv'))
mu_val = binDataframe(mu_val)
mu_val.to_csv("data/mu_val.csv")
gc.collect()

print('Loading mu hidden set...')
mu_hidden = pd.read_csv(os.path.join('data', 'mu_hidden.csv'))
mu_hidden = binDataframe(mu_hidden)
mu_hidden.to_csv("data/mu_hidden.csv")
gc.collect()

print('Loading mu qual set...')
mu_qual = pd.read_csv(os.path.join('data', 'mu_qual.csv'))
mu_qual = binDataframe(mu_qual)
mu_qual.to_csv("data/mu_qual.csv")
gc.collect()

print('Loading mu train set...')
mu_train = pd.read_csv(os.path.join('data', 'mu_train.csv'))
mu_train = binDataframe(mu_train)
mu_train.to_csv("data/mu_train.csv")
gc.collect()
