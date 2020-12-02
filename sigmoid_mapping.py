# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:27:31 2020

@author: user
"""

import numpy as np



#euclidean distance
def euclidean_dist(a,b):
    result = 0
    for i in zip(a,b):
        result += (i[0] - i[1])**2
        
    return np.sqrt(result)



# change rating to sigmoid_mapping fuction vlaue
# ratings_list: [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5] for movielens dataset.
# mid means rating which is 'so-so'
ratings_list = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
def sigmoid_mapping_d(ratings_list = ratings_list, mid = 3.5): 
    
    sigmoid_dic = {}
    
    for i in ratings_list:
        sigmoid_dic[i] = (1 + mid**2) / (1 + np.exp(-i + mid))
    
    return sigmoid_dic

# distance between sigmoid mapping values
def sigmoid_mapping_similarity(ui, uj, sigmoid_dic):
    
    if len(ui) == 0:
        return 0
    
    dist = []

    for i, j in zip(ui, uj):
        a = (i, sigmoid_dic[i])
        b = (j, sigmoid_dic[j])
        
        dist.append(euclidean_dist(a,b))

    similarity = 1/abs(1+sum(dist))

    return similarity
    
    
    

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
