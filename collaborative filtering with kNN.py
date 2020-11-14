# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:16:09 2020

@author: user
"""

#%%
import load_data
import similarity_methods
import sigmoid_mapping

from scipy.sparse import csr_matrix
from tqdm import tqdm
import numpy as np
import random

#%%

# 1. data load
rating_matrix = load_data.create_rating()

# np.mean(rating_matrix ['rating']) # 3.50
# np.median(rating_matrix ['rating']) # 3.5
# np.mean(list(Counter(rating_matrix ['userId']).values())) # 평균 165개의 rating을 하였음.
# np.min(list(Counter(rating_matrix ['userId']).values())) # 최솟값 20개
# np.max(list(Counter(rating_matrix ['userId']).values())) # 최댓값 2698개

#%%

# 2. split train / test set 

def train_test_split(data, test_ratio = 0.2): # data: movie lens dataset,

    train_user_idx = [] # train row
    train_item_idx = [] # train columns
    train_rating_idx = [] # train ratings
    test_user_idx = [] # test row
    test_item_idx = [] # test columns
    test_rating_idx = [] # test ratings
    
    
    user_length = len(set(data['userId'])) # length of users
    for i in range(1,user_length+1):
        
        sub = rating_matrix[data['userId']==i]
        index = list(sub.index)
        random.shuffle(index)    
        cutting = int(len(index)*test_ratio)
        
        train_idx = index[cutting:] # train index of subset
        test_idx = index[:cutting]  # test index of subset
        
        train_user_idx.extend(np.repeat(i,len(train_idx)))
        train_item_idx.extend(sub.loc[train_idx, 'movieId'])
        train_rating_idx.extend(sub.loc[train_idx, 'rating'])
        
        test_user_idx.extend(np.repeat(i,len(test_idx)))
        test_item_idx.extend(sub.loc[test_idx, 'movieId'])
        test_rating_idx.extend(sub.loc[test_idx, 'rating'])
        
    
    if len(train_user_idx) == len(train_item_idx):
        train_csr = csr_matrix((train_rating_idx, (train_user_idx, train_item_idx)))
    
    if len(test_user_idx) == len(test_item_idx):
        test_csr = csr_matrix((test_rating_idx, (test_user_idx, test_item_idx)))

    return train_csr, test_csr

train_csr_matrix, test_csr_matrix = train_test_split(rating_matrix, test_ratio = 0.2)


#%%

# 3. similarity calculation




length = list(range(1, train_csr_matrix.shape[0])) # user length


similarity_measure = 'cosine_similarity'
if similarity_measure == 'cosine_similarity':
    
    cos_sim_dic = {}
    
    for i in tqdm(length): # active users loop
        
        cos_sim_dic[i] = {}
    
        neighbor_ = length.copy()
        neighbor_.remove(i)
        
        for j in neighbor_: # neighbor except for active user
            active_user_item = set(train_csr_matrix[i,:].indices) # active user's ratings
            co_rated = list(active_user_item.intersection(set(train_csr_matrix[j,:].indices))) # co-rated rating
            active = train_csr_matrix[i,co_rated].data # active user's ratings in co-rated
            neighbor = train_csr_matrix[j,co_rated].data # neighbor user's ratings in co-rated
            
            sim = similarity_methods.cosine_similarity(active, neighbor)
            
            cos_sim_dic[i][j] = [sim,len(co_rated),j] # similarity, count co-rated, neighbor



similarity_measure =  'PCC_similarity'
if similarity_measure == 'PCC_similarity':
    
    pcc_sim_dic = {}
    
    for i in tqdm(length): # active users loop
        
        pcc_sim_dic[i] = {}
    
        neighbor_ = length.copy()
        neighbor_.remove(i)
        
        for j in neighbor_: # neighbor except for active user
            active_user_item = set(train_csr_matrix[i,:].indices) # active user's ratings
            co_rated = list(active_user_item.intersection(set(train_csr_matrix[j,:].indices))) # co-rated rating
            active = train_csr_matrix[i,co_rated].data # active user's ratings in co-rated
            neighbor = train_csr_matrix[j,co_rated].data # neighbor user's ratings in co-rated
            
            sim = similarity_methods.PCC_similarity(active, neighbor)
            
            pcc_sim_dic[i][j] = [sim,len(co_rated),j] # similarity, count co-rated, neighbor            
 
 
 
 
similarity_measure =  'MSD_similarity'
if similarity_measure == 'MSD_similarity':
    
    msd_sim_dic = {}
    
    for i in tqdm(length): # active users loop
        
        msd_sim_dic[i] = {}
    
        neighbor_ = length.copy()
        neighbor_.remove(i)
        
        for j in neighbor_: # neighbor except for active user
            active_user_item = set(train_csr_matrix[i,:].indices) # active user's ratings
            co_rated = list(active_user_item.intersection(set(train_csr_matrix[j,:].indices))) # co-rated rating
            active = train_csr_matrix[i,co_rated].data # active user's ratings in co-rated
            neighbor = train_csr_matrix[j,co_rated].data # neighbor user's ratings in co-rated
            
            sim = similarity_methods.MSD_similarity(active, neighbor)
            
            msd_sim_dic[i][j] = [sim,len(co_rated),j] # similarity, count co-rated, neighbor 




similarity_measure =  'Jaccard_similarity'
if similarity_measure == 'Jaccard_similarity':
    
    jac_sim_dic = {}
    
    for i in tqdm(length): # active users loop
        
        jac_sim_dic[i] = {}
    
        neighbor_ = length.copy()
        neighbor_.remove(i)
        
        for j in neighbor_: # neighbor except for active user
            active_user_item = set(train_csr_matrix[i,:].indices) # active user's ratings
            co_rated = list(active_user_item.intersection(set(train_csr_matrix[j,:].indices))) # co-rated rating
            active = train_csr_matrix[i,co_rated].data # active user's ratings in co-rated
            neighbor = train_csr_matrix[j,co_rated].data # neighbor user's ratings in co-rated
            
            sim = similarity_methods.Jaccard_similarity(train_csr_matrix[i,:].indices, 
                                               train_csr_matrix[j,:].indices)
            
            jac_sim_dic[i][j] = [sim,len(co_rated),j] # similarity, count co-rated, neighbor 






similarity_measure =  'sigmoid_mapping_similarity'   
if similarity_measure == 'sigmoid_mapping_similarity':
    
    sigmoid_dic = sigmoid_mapping.sigmoid_mapping_d()
    
    sig_sim_dic = {}
    
    for i in tqdm(length): # active users loop
        
        sig_sim_dic[i] = {}
    
        neighbor_ = length.copy()
        neighbor_.remove(i)
        
        for j in neighbor_: # neighbor except for active user
            active_user_item = set(train_csr_matrix[i,:].indices) # active user's ratings
            co_rated = list(active_user_item.intersection(set(train_csr_matrix[j,:].indices))) # co-rated rating
            active = train_csr_matrix[i,co_rated].data # active user's ratings in co-rated
            neighbor = train_csr_matrix[j,co_rated].data # neighbor user's ratings in co-rated
            
            sim = sigmoid_mapping.sigmoid_mapping_similarity(active, neighbor, 
                                                     sigmoid_dic = sigmoid_dic)
            
            sig_sim_dic[i][j] = [sim,len(co_rated),j] # similarity, count co-rated, neighbor 
            
            
    
#%%
# make simliarity matrics to pickle   
import pickle 
dics = [cos_sim_dic, pcc_sim_dic, msd_sim_dic, jac_sim_dic, sig_sim_dic]
dics_n = ['cos_sim_dic', 'pcc_sim_dic', 'msd_sim_dic', 'jac_sim_dic', 'sig_sim_dic']

for i,n in zip(dics, dics_n):
    with open('%s.pickle'.format(n), 'wb') as f:
        pickle.dump(i, f)
    

#%%

def user_item_dictionary(data):
    
    rating_dic = {}
    
    for user in set(rating_matrix['userId']):
        
        rating_dic[user] = {}
        it = rating_matrix[rating_matrix['userId']==user][['movieId','rating']]
        
        for item, rating in zip(it['movieId'], it['rating']):
            
            rating_dic[user][item] = rating
    return rating_dic
            
rd = user_item_dictionary(rating_matrix)
len(rd)





items = set(rating_matrix['movieId'])

unrated = items.difference()

#%%

# prediction KNN 


#%%

# performance check
