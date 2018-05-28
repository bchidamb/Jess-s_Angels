import pandas as pd
import numpy as np
import gc

probe_size = 100000

###
# MU data
###

mu_data = pd.read_csv("mu/all.dta", sep=' ',header=None)
mu_data.columns = ["User Number", "Movie Number", "Date Number", "Rating"]
print("Full mu data")

mu_idx = pd.read_table("mu/all.idx", header=None)
mu_idx.columns = ["Index"]
print("Full mu indexing")

four = mu_idx.loc[mu_idx["Index"] == 4]
four = four.index.tolist()


real_probe_indices = np.random.choice(four, probe_size, replace=False)
probe_set = mu_data.loc[real_probe_indices].copy()

probe_set.to_csv("data/real_mu_probe.csv")
probe_set.drop(["Date Number"], axis=1).to_csv("data/real_mu_probe.txt", header=None, index=False, sep=' ')
print(len(probe_set))

one = mu_idx.loc[mu_idx["Index"] == 1]
train_list = one.index.tolist()
two = mu_idx.loc[mu_idx["Index"] == 2]
train_list += two.index.tolist()
three = mu_idx.loc[mu_idx["Index"] == 3]
train_list += three.index.tolist()

train_list += list(set(four) - set(real_probe_indices))

mu_train = mu_data.loc[train_list]
mu_train.to_csv("data/real_mu_train.csv")
mu_train.drop(["Date Number"], axis=1).to_csv("data/real_mu_train.txt", header=None, index=False, sep=' ')
print(len(mu_train))

gc.collect()

###
# um data
###

um_data = pd.read_csv("um/all.dta", sep=' ',header=None)
um_data.columns = ["User Number", "Movie Number", "Date Number", "Rating"]
print("Full um data")

um_idx = pd.read_table("um/all.idx", header=None)
um_idx.columns = ["Index"]
print("Full um indexing")

four = um_idx.loc[um_idx["Index"] == 4]
four = four.index.tolist()

real_probe_indices = np.random.choice(four, probe_size, replace=False)
probe_set = um_data.loc[real_probe_indices].copy()

probe_set.to_csv("data/real_um_probe.csv")
probe_set.drop(["Date Number"], axis=1).to_csv("data/real_um_probe.txt", header=None, index=False, sep=' ')

print(len(probe_set))

one = um_idx.loc[um_idx["Index"] == 1]
train_list = one.index.tolist()
two = um_idx.loc[um_idx["Index"] == 2]
train_list += two.index.tolist()
three = um_idx.loc[um_idx["Index"] == 3]
train_list += three.index.tolist()

train_list += list(set(four) - set(real_probe_indices))

um_train = um_data.loc[train_list]
um_train.to_csv("data/real_um_train.csv")
um_train.drop(["Date Number"], axis=1).to_csv("data/real_um_train.txt", header=None, index=False, sep=' ')
print(len(um_train))
