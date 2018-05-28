
# coding: utf-8

# In[3]:


import numpy as np
from tqdm import *
import heapq
import math


# In[2]:


num_q = 150 # Number of users to check knn with.
num_k = 50 # Number of neighbors to use. 
train_path = 'mu/all.dta'
test_path = 'mu/qual.dta'
f = open(train_path)
users = {}
counter = 0
test = 1e20
for line in tqdm(f):
    counter += 1
    data = np.fromstring(line, dtype=int, sep=' ')
    userId = data[0]
    movieId = data[1]
    rating = data[3]
    if rating == 0: # Rating blanked out.
        continue 
    if counter > test:
        break
    if userId in users:
        users[userId][movieId] = rating
    else:
        users[userId] = {movieId: rating}
f.close()


# In[3]:


print("Averaging scores")
averages = {}
for userId in tqdm(users):
    averages[userId] = sum([score for score in users[userId].values()]) * 1.0 / len(users[userId])


# In[4]:


def calc_closeness(userId1, userId2):
    # Pearson correlation coefficient
    num_shared_movies = 0
    cov = 0
    sd1 = 0
    sd2 = 0
    avg1 = averages[userId1]
    avg2 = averages[userId2]

    for movieId in users[userId1]:
        if movieId in users[userId2]:
            num_shared_movies += 1
            r1 = users[userId1][movieId]
            r2 = users[userId2][movieId]
            d1 = r1 - avg1
            d2 = r2 - avg2
            cov += d1 * d2
            sd1 += d1 ** 2
            sd2 += d2 ** 2
    if num_shared_movies and sd1 > 0 and sd2 > 0:
        return (1.0 / num_shared_movies * cov / (math.sqrt(1.0 / num_shared_movies * sd1) * math.sqrt(1.0 / num_shared_movies * sd2)))
    return 0.0


# In[ ]:


Q = []
print("Finding top q neighbors")
for k, v in tqdm(users.items()):
    if len(Q) == num_q:
        if Q[-1][0] < len(v):
            heapq.heappushpop(Q, (len(v), k))
    else:
        heapq.heappush(Q, (len(v), k))

Q = [id[1] for id in Q]

knn_table = {}
print("Generating knn from top-q")
for userId in tqdm(users):
    top_k = []
    for qId in Q:
        closeness = calc_closeness(userId, qId)
        if len(top_k) == num_k:
            if top_k[-1][0] < closeness:
                heapq.heappushpop(top_k, (closeness, qId))
        else:
            heapq.heappush(top_k, (closeness, qId))
            
    knn_table[userId] = top_k

def predict(userId, movieId):
    totalWeight = 0
    rating = 0
    if userId not in users:
        return 0
    avg1 = averages[userId]

    for neighbor in knn_table[userId]:
        # If neighbor id has movie id
        if movieId in users[neighbor[1]]:
            avg2 = averages[neighbor[1]]
            totalWeight += neighbor[0]
            rating += (users[neighbor[1]][movieId] - avg2) * neighbor[0]
    if totalWeight:
        rating /= totalWeight
    rating += avg1
    return rating



# In[2]:


def predict(userId, movieId):
    totalWeight = 0
    rating = 0
    if userId not in users:
        return 0
    avg1 = averages[userId]

    '''for neighbor in knn_table[userId]:
        # If neighbor id has movie id
        if movieId in users[neighbor[1]]:
            avg2 = averages[neighbor[1]]
            totalWeight += neighbor[0]
            rating += (users[neighbor[1]][movieId] - avg2) * neighbor[0]
    if totalWeight:
        rating /= totalWeight
    rating += avg1
    '''
    print(avg1)

# Load test set
test_path = 'mu/qual.dta'
print("Loading test data")
f = open(test_path)
test_data = []
for line in tqdm(f):
    data = np.fromstring(line, dtype=int, sep=' ')
    userId = data[0]
    movieId = data[1]
    test_data.append([ userId, movieId])
f.close()

# Predict on test set, and write to file
print("Writing predictions")
predictions = [predict(test_point[0], test_point[1]) for test_point in tqdm(test_data)]
print("Saving file")
np.savetxt("%dQ_%dKnn" %(num_q, num_k), np.array(predictions),
           fmt ='%1.5f')


# In[ ]:


num_q = 150 # Number of users to check knn with.
num_k = 50 # Number of neighbors to use. 
train_path = 'mu/all.dta'
test_path = 'mu/qual.dta'
f = open(train_path)
users = {}
counter = 0
test = 1e15
for line in tqdm(f):
    counter += 1
    data = np.fromstring(line, dtype=int, sep=' ')
    userId = data[0]
    movieId = data[1]
    rating = data[3]
    if rating == 0:
        continue 
    if counter > test:
        break
    if userId in users:
        users[userId][movieId] = rating
    else:
        users[userId] = {movieId: rating}
f.close()

averages = {}
for userId in users:
    averages[userId] = sum([score for score in users[userId].values()]) * 1.0 / len(users[userId])

