# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 17:09:52 2020

@author: user
"""
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, csc_matrix
from load_data import load_data
import pandas as pd
import numpy as np


rd = load_data.user_item_dictionary()
rd_1m = load_data.user_item_dictionary_1M()

users=[]
items=[]
ratings=[]

for user in rd_1m:
    for item in rd_1m[user]:
        users.append(user)
        items.append(item)
        ratings.append(rd_1m[user][item])

rating_matrix = csr_matrix((ratings, (users, items)))

global_mean = np.mean(ratings)

# 유사도 행렬을 만드는 과정.

similarity_matrix_sk = cosine_similarity(rating_matrix)

df = pd.DataFrame(similarity_matrix)

# user 별로 돌고, 또 user 별로 돌면서 corated를 찾으면서, similarity 를 계산한다. 

for i in :
csr_matrix.