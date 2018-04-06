# Puts data into respective train, validation, hidden, probe, and qual sets

import pandas as pd
import numpy as np

###
# MU data
###

mu_data = pd.read_csv("mu/all.dta", sep=' ',header=None)
mu_data.columns = ["User Number", "Movie Number", "Date Number", "Rating"]
print("Full mu data")

mu_idx = pd.read_table("mu/all.idx", header=None)
mu_idx.columns = ["Index"]
print("Full mu indexing")

# Get the indices of the rows of the respective datasets
one = mu_idx.loc[mu_idx["Index"] == 1]
one = one.index.tolist()

two = mu_idx.loc[mu_idx["Index"] == 2]
two = two.index.tolist()

three = mu_idx.loc[mu_idx["Index"] == 3]
three = three.index.tolist()

four = mu_idx.loc[mu_idx["Index"] == 4]
four = four.index.tolist()

five = mu_idx.loc[mu_idx["Index"] == 5]
five = five.index.tolist()
print("Broke up indices")

mu_train = mu_data.loc[one]
mu_val = mu_data.loc[two]
mu_hidden  = mu_data.loc[three]
mu_probe = mu_data.loc[four]
mu_qual = mu_data.loc[five]
print("Got indexed items in separate dataframes")

mu_train.to_csv("data/mu_train.csv")
mu_val.to_csv("data/mu_val.csv")
mu_hidden.to_csv("data/mu_hidden.csv")
mu_probe.to_csv("data/mu_probe.csv")
mu_qual.to_csv("data/mu_qual.csv")

print("Loaded all mu data")
###
# UM data
###

um_data = pd.read_csv("um/all.dta", sep=' ',header=None)
um_data.columns = ["User Number", "Movie Number", "Date Number", "Rating"]
print("Full um data")

um_idx = pd.read_table("um/all.idx", header=None)
um_idx.columns = ["Index"]
print("Full um indexing")

# Get the indices of the rows of the respective datasets
um_one = um_idx.loc[um_idx["Index"] == 1]
um_one = um_one.index.tolist()

um_two = um_idx.loc[um_idx["Index"] == 2]
um_two = um_two.index.tolist()

um_three = um_idx.loc[um_idx["Index"] == 3]
um_three = um_three.index.tolist()

um_four = um_idx.loc[um_idx["Index"] == 4]
um_four = um_four.index.tolist()

um_five = um_idx.loc[um_idx["Index"] == 5]
um_five = um_five.index.tolist()
print("Broke up indices")

um_train = um_data.loc[um_one]
um_val = um_data.loc[um_two]
um_hidden  = um_data.loc[um_three]
um_probe = um_data.loc[um_four]
um_qual = um_data.loc[um_five]
print("Got indexed items in separate dataframes")

um_train.to_csv("data/um_train.csv")
um_val.to_csv("data/um_val.csv")
um_hidden.to_csv("data/um_hidden.csv")
um_probe.to_csv("data/um_probe.csv")
um_qual.to_csv("data/um_qual.csv")

print("Loaded all um data")
