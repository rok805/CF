# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:29:58 2020

@author: user
"""


from load_data import load_data
from similarity_measure import similarity_methods, sigmoid_mapping
from tqdm import tqdm
import numpy as np
import random
import pickle


# 1. data load
rd = load_data.user_item_dictionary()


# 2. train/test split
def train_test_split(data, test_ratio = 0.2, random_state = 92): # data: movie lens dataset
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

train, test = train_test_split(data=rd, test_ratio = 0.2)




def similarity_calculation(data, measure = 'cosine'):
    print('similarity calculation')
    
    user_id = list(data.keys())
    sim_dic = {}
    idx = 0
    
    if measure == 'cosine':
        sim_measure = similarity_methods.cosine_similarity
        idx+=1
    elif measure == 'pcc':
        sim_measure = similarity_methods.PCC_similarity
        idx+=1
    elif measure == 'msd':
        sim_measure = similarity_methods.MSD_similarity
        idx+=1
    elif measure == 'jaccard':
        sim_measure = similarity_methods.Jaccard_similarity
        idx+=2
    elif measure == 'sigmoid':
        sim_measure = sigmoid_mapping.sigmoid_mapping_similarity
        idx+=3
    
    
    if idx == 1: # For cosine, pcc, msd 
        for user in tqdm(user_id):
            
            sim_dic[user] = {}
            
            tmp = user_id.copy()
            tmp.remove(user)
            neighbors = tmp # active user removed.
            
            for neighbor in neighbors:
                
                corated = set(data[user].keys()).intersection(set(data[neighbor].keys()))
                
                ui = []
                uj = []
                
                for c in corated:
                    
                    ui.append(data[user][c])
                    uj.append(data[neighbor][c])
                
                sim_dic[user][neighbor] = (sim_measure(ui, uj), len(corated))
        
        return sim_dic
    
    
    if idx == 2: # For jaccard
        for user in tqdm(user_id):
            
            sim_dic[user] = {}
            
            tmp = user_id.copy()
            tmp.remove(user)
            neighbors = tmp # active user removed.
            
            for neighbor in neighbors:
                
                ui = data[user].keys()
                uj = data[neighbor].keys()
                
                corated = set(ui).intersection(set(uj))
                
                sim_dic[user][neighbor] = (sim_measure(ui,uj), len(corated))
        
        return sim_dic
    
    
    if idx == 3: # For sigmoid
        
        sigmoid_dic = sigmoid_mapping.sigmoid_mapping_d()
        
        for user in tqdm(user_id):
            
            sim_dic[user] = {}
            
            tmp = user_id.copy()
            tmp.remove(user)
            neighbors = tmp # active user removed.
            
            for neighbor in neighbors:
                
                corated = set(data[user].keys()).intersection(set(data[neighbor].keys()))
                
                ui = []
                uj = []
                
                for c in corated:
                    
                    ui.append(data[user][c])
                    uj.append(data[neighbor][c])
                
                sim_dic[user][neighbor] = (sim_measure(ui, uj, sigmoid_dic = sigmoid_dic), len(corated))
        
        return sim_dic





# uses similarity calculation
cosine_sim = similarity_calculation(train, 'cosine')
pcc_sim = similarity_calculation(train,'pcc')
msd_sim = similarity_calculation(train,'msd')
jaccard_sim = similarity_calculation(train,'jaccard')
sigmoid_sim = similarity_calculation(train,'sigmoid')


# make simliarity matrics to pickle file
dics = [cosine_sim, pcc_sim, msd_sim, jaccard_sim, sigmoid_sim]
dics_n = ['cos_sim_dic', 'pcc_sim_dic', 'msd_sim_dic', 'jac_sim_dic', 'sig_sim_dic']



def predict_with_knn(data, sim_metric=cosine_sim, k=10):
    print('predict')
    
    k=7
    rating = load_data.create_rating()
    items = set(rating['movieId'])
    
    avg_d = {user:np.mean(list(data[user].values())) for user in data}
    predict_d = {}
    users = data.keys()
    
    for ui in tqdm((users), position=0, leave=True):
        
        unrated = items.difference(set(data[ui].keys()))
        active_item_dic = {}
        active_item_dic[ui] = {}
        predict_d[ui] = {}
        
        #1. find 'active item' rated by 'nearest neighbor'    
        for active_item in unrated:
            active_item_dic[ui][active_item]=[]
            
            for neighbor in users:
                if active_item in data[neighbor].keys(): # neighbor 중에서 unrated active item을 갖고 있다면,
                    active_item_dic[ui][active_item].append(sim_metric[ui][neighbor] + (data[neighbor][active_item],)) # similarity, number of co-rated, rating        
        
        #2. create predict value
            predict_d[ui][active_item] = 0
            
            k_near_neighbor = sorted(active_item_dic[ui][active_item], key=lambda x: (x[0],[1]), reverse=True)[:k]
            
            up = 0
            down = 0
            
            for i in k_near_neighbor:
                up += (i[0]*i[2])
                down += i[0]
            try:
                predict_d[ui][active_item] = up/down    
            except ZeroDivisionError:
                predict_d[ui][active_item] = avg_d[ui] # error 발생시 mean값으로 대체.
    
    return predict_d


predict_cos = predict_with_knn(data = train, sim_metric = cosine_sim, k = 10)
predict_pcc = predict_with_knn(data = train, sim_metric = pcc_sim, k = 10)
predict_msd = predict_with_knn(data = train, sim_metric = msd_sim, k = 10)
predict_jac = predict_with_knn(data = train, sim_metric = jaccard_sim, k = 10)
predict_sig = predict_with_knn(data = train, sim_metric = sigmoid_sim, k = 10)



def MAE(pred, real):
    
    up = sum([abs(i[0]-i[1]) for i in zip(pred, real)])
    down = len(pred)
    
    return up/down


def performance(data, test, predict_d):
    print('performance')
    users = data.keys() # whole users
    
    performance_mae = 0
    
    for i in users: # 모든 유저를 돌면서, 
        pred_=[]
        real_=[]
        for j in test[i]: # test 셋이 존재하는 item에 대해,
            pred_.append(predict_d[i][j])
            real_.append(test[i][j])
        
        performance_mae += MAE(pred_, real_)
    
    return performance_mae / len(users)

perform_cos = performance(data=rd, test=test, predict_d=predict_cos)
perform_pcc = performance(data=rd, test=test, predict_d=predict_pcc)
perform_msd = performance(data=rd, test=test, predict_d=predict_msd)
perform_jac = performance(data=rd, test=test, predict_d=predict_jac)
perform_sig = performance(data=rd, test=test, predict_d=predict_sig)