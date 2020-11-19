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




# similarity_calculation
def similarity_calculation(data, measure):
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
    elif measure == 'sigmoid_jaccard':
        sim_measure = sigmoid_mapping.sigmoid_mapping_similarity
        idx+=4
    
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


    if idx == 4: # For sigmoid with jaccard
    
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
                
                s = sim_measure(ui, uj, sigmoid_dic = sigmoid_dic) #sig
                j = len(corated) / (len(data[user].keys()) + len(data[neighbor].keys())) #jaccard
                
                sim_dic[user][neighbor] = (s*j, len(corated))
        
        return sim_dic




# prediction
def predict_with_knn(data, sim_metric, k):
    print('predict')
    
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
        # return values = similarity, number of co-rated, rating
        for active_item in unrated:
            active_item_dic[ui][active_item]=[]
            
            for neighbor in users:
                if active_item in data[neighbor].keys(): 
                    active_item_dic[ui][active_item].append(sim_metric[ui][neighbor] + (data[neighbor][active_item],))      
        
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
                if np.isnan(predict_d[ui][active_item]):
                    predict_d[ui][active_item] = avg_d[ui]
            except ZeroDivisionError:
                predict_d[ui][active_item] = avg_d[ui] # replace mean when error appears.
    
    return predict_d




# MAE formula
def MAE(pred, real):
    
    up = sum([abs(i[0]-i[1]) for i in zip(pred, real)])
    down = len(pred)
    
    return up/down




# performance calculation
def performance_mae(data, test, predict_d):
    print('performance_mae')
    users = data.keys() # whole users
    
    performance_mae = 0
    
    for i in users: 
        pred_=[]
        real_=[]
        for j in test[i]: 
            pred_.append(predict_d[i][j])
            real_.append(test[i][j])
        
        performance_mae += MAE(pred_, real_)
    
    return performance_mae / len(users)






# 1. data load
rd = load_data.user_item_dictionary()    

performance_d = {}


kf_=3
knn_=[5,10,15,20]

for k_fold in list(range(kf_)):
    print('k_fold {}'.format(k_fold))
    performance_d[k_fold] = {}
    
    # 2. split train/test set
    train, test = train_test_split(data=rd, test_ratio = 0.2, random_state = k_fold)

    # 3. uses similarity calculation
    sim_cos = similarity_calculation(train, 'cosine')
    sim_pcc = similarity_calculation(train,'pcc')
    sim_msd = similarity_calculation(train,'msd')
    sim_jac = similarity_calculation(train,'jaccard')
    sim_sig = similarity_calculation(train,'sigmoid')
    sim_sig2 = similarity_calculation(train,'sigmoid_jaccard')

    
    for k_neighbor in knn_:
        
        # 4. prediction
        predict_cos = predict_with_knn(data = train, sim_metric = sim_cos, k = k_neighbor)
        predict_pcc = predict_with_knn(data = train, sim_metric = sim_pcc, k = k_neighbor)
        predict_msd = predict_with_knn(data = train, sim_metric = sim_msd, k = k_neighbor)
        predict_jac = predict_with_knn(data = train, sim_metric = sim_jac, k = k_neighbor)
        predict_sig = predict_with_knn(data = train, sim_metric = sim_sig, k = k_neighbor)
        predict_sig2 = predict_with_knn(data = train, sim_metric = sim_sig2, k = k_neighbor)
    
    
        # 5. performance
        mae_cos = performance_mae(data=rd, test=test, predict_d=predict_cos)
        mae_pcc = performance_mae(data=rd, test=test, predict_d=predict_pcc)
        mae_msd = performance_mae(data=rd, test=test, predict_d=predict_msd)
        mae_jac = performance_mae(data=rd, test=test, predict_d=predict_jac)
        mae_sig = performance_mae(data=rd, test=test, predict_d=predict_sig)
        mae_sig2 = performance_mae(data=rd, test=test, predict_d=predict_sig2)

        
        performance_d[k_fold][k_neighbor]={'cos':mae_cos,
                                           'pcc':mae_pcc,
                                           'msd':mae_msd,
                                           'jac':mae_jac,
                                           'sig':mae_sig,
                                           'sig2':mae_sig2}


