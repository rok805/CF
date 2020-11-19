# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 23:26:44 2020

@author: user
"""

import numpy as np
num_sim_user_topk = 2
num_item_rec_topk = 2
num_users = 5
num_items = 10
num_rows = 15
rating = np.array([
    [1, 1, 1.0],
    [1, 2, 2.0],
    [1, 5, 1.2],
    [2, 2, 1.5],
    [2, 3, 3.0],
    [3, 1, 2.2],
    [3, 2, 6.2],
    [3, 7, 1.5],
    [4, 6, 1.2],
    [4, 3, 1.5],
    [4, 1, 3.1],
    [4, 2, 4.0],
    [5, 4, 8.2],
    [5, 2, 6.5],
    [5, 7, 8.0]
    ]
)

# similarity measure


# rating matrix를 dictionary로 만듬.
rating_d={}
for i in rating:
    if i[0] in rating_d.keys():
        rating_d[int(i[0])][int(i[1])] = i[2]
    else:
        rating_d[int(i[0])] = {}
        rating_d[int(i[0])][int(i[1])] = i[2]



def pcc(x,y,x_mean,y_mean):
    
    up = sum([(i-x_mean)*(j-y_mean) for i,j in zip(x,y)])
    d1 = sum([(i-x_mean)**2 for i in x])
    d2 = sum([(i-y_mean)**2 for i in y])
    down = np.sqrt(d1) * np.sqrt(d2)

    return up/down

# 유사도 계산
sim_d = {}
for i in rating_d.keys():
    
    neighbors = list(rating_d.keys())
    neighbors.remove(i)
    
    sim_d[i]={}
    
    for j in neighbors:

        ui = set(rating_d[i].keys())
        uj = set(rating_d[j].keys())
        corated = list(ui.intersection(uj))
        
        ui_r = list(rating_d[i].values())
        uj_r = list(rating_d[j].values())
        ui_mean = np.mean(ui_r)
        uj_mean = np.mean(uj_r)
        
        ui_c = [rating_d[i][it] for it in rating_d[i].keys() if it in corated]
        uj_c = [rating_d[j][it] for it in rating_d[j].keys() if it in corated]
        
        sim = pcc(ui_c, uj_c, ui_mean, uj_mean)
        
        sim_d[i][j] = sim
    
    sim_d[i] = sorted(sim_d[i].items(), key = lambda x: x[1], reverse = True)
    


# recommend
recommend_d = {}
items = set([int(i[1]) for i in rating])

for i in [1,2,3,4,5]:

    recommend_d[i] = {} # ui의 추천리스트
    
    ui_mean = np.mean(list(rating_d[i].values()))
    
    unrated = items.difference(set(rating_d[i].keys()))
    
    for j in unrated:
        
        recommend_d[i][j] = 0
        
        up = []
        down = []
        idx = 0
        
        for nei, s in sim_d[i]:
            uj_mean = np.mean(list(rating_d[nei].values()))
            
            if j in rating_d[nei].keys():
                up.append(s * (rating_d[nei][j] - uj_mean) + ui_mean)
                down.append(abs(s))
                idx+=1
            else:
                continue
            
            if idx == num_sim_user_topk:
                break
            
        try:
            recommend_d[i][j] = sum(up)/len(up)
        except ZeroDivisionError:
            pass
                
        
   
num_reco_users = 2 # (추천결과를 만들어야 할 유저수)
rec_1 = 1
rec_2 = 2
sorted(recommend_d[rec_1].items(), key=lambda x: x[1], reverse=True)[:num_reco_users]
sorted(recommend_d[rec_2].items(), key=lambda x: x[1], reverse=True)[:num_reco_users]
