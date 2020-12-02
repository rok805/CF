# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 17:28:17 2020

@author: user
"""



from load_data import load_data
from similarity_measure import similarity_methods, sigmoid_mapping
from tqdm import tqdm
import numpy as np
import random




# train/test split
def train_test_split(data, test_ratio = 0.2, random_state = 1004): # data: movie lens dataset
    print('split dataset')    
    train_set = {}
    test_set = {}
    
    for user in data:
        
        item_rate = list(data[user].items())
        random.Random(random_state).shuffle(item_rate)
        length = len(item_rate)
        cri = int(length * (1-0.2))
        
        train_set[user] = {i:r for i,r in item_rate[:cri]}
        test_set[user] = {i:r for i,r in item_rate[cri:]}
    
    return train_set, test_set


rd = load_data.user_item_dictionary()
train, test = train_test_split(data=rd, test_ratio=0.2, random_state=1004)


def neighbor(data, measure = 'cosine'):
    
    users = rd.keys()
    
    for ui in users:
        
    