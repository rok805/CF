#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 04:17:20 2020

@author: cheongrok
"""


from similarity_measure import similarity_methods, sigmoid_mapping
from load_data import load_data
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import datetime
import random
import pickle
import math
import copy
import time

class CF:

    def __init__(self, data, test_ratio, CV, measure, k, soso=3, new=0):
        self.data = data
        self.new_data = None
        self.test_ratio = test_ratio
        self.CV = CV
        self.cv = 0
        self.measure = measure
        self.k = k
        self.mid = soso
        self.new = new

        #  for rating sigmoid mapping.
        self.ratings_list = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
        
        self.max_r = max(self.ratings_list)



    def cv_div(self,x):
        idx = list(range(len(x)))
        div = len(x) // self.CV + 1
        div_mok = len(idx) // self.CV
        can = []

        if div-div_mok > 0:
            over = len(idx) - div_mok * 5

        for i in range(over):
            tmp=[]
            for j in range(div_mok + 1):
                tmp.append(idx.pop())
            can.append(tmp) 

        while idx:
            tmp=[]
            for i in range(div_mok):
                tmp.append(idx.pop())
            can.append(tmp)
        return can


    # train/test split
    def train_test_split(self):
        
        #  for new_rating_mean method
        self.max_r_new = set()

        print('--------------------train test split-------------------------')
        print()

        self.train = {}
        self.test = {}

        for user in self.data:

            item_rate = list(self.data[user].items())
            random.Random(7777).shuffle(item_rate)
            length = len(item_rate)

            basket = self.cv_div(item_rate)[self.cv]

            self.train[user] = {item_rate[i][0]:item_rate[i][1] for i in range(length) if i not in basket}
            self.test[user] = {item_rate[i][0]:item_rate[i][1] for i in range(length) if i in basket}

        self.cv+=1

        # the number of ratings in train set. for sim_PNCR
        self.N = []
        for i in self.train:
            self.N.extend(list(self.train[i].keys()))
        self.N = len(set(self.N))

        # mean of train values
        if self.mid == 'mean':
            self.mid = []
            for i in self.train:
                self.mid.extend(list(self.train[i].values()))
            self.mid = np.mean(self.mid)


        #  for new rating. 각 user의 평균을 사용하여 rating change. 1
        if self.new == 1:
            self.new_data = copy.deepcopy(self.train)
            self.new_data = sigmoid_mapping.new_rating_mean_1(self.new_data)  # new rating method
    
            for i in self.new_data:
                for j in self.new_data[i]:
                    self.max_r_new.add(self.new_data[i][j])
            self.max_r_new = max(self.max_r_new)

        if self.new == 1.5:
            self.new_data = copy.deepcopy(self.train)
            self.new_data = sigmoid_mapping.new_rating_mean_1_1(self.new_data)  # new rating method

            for i in self.new_data:
                for j in self.new_data[i]:
                    self.max_r_new.add(self.new_data[i][j])
            self.max_r_new = max(self.max_r_new)

        #  for new rating. 각 user의 평균을 사용하여 rating change. 2
        elif self.new == 2:
            self.new_data = copy.deepcopy(self.train)
            self.new_data = sigmoid_mapping.new_rating_mean_sigmoid_2(self.new_data)  # new rating method
    
            for i in self.new_data:
                for j in self.new_data[i]:
                    self.max_r_new.add(self.new_data[i][j])
            self.max_r_new = max(self.max_r_new)
        
        #  for new rating. 각 user의 평균과 분산을 사용하여 rating change. 3
        elif self.new == 3:
            self.new_data = copy.deepcopy(self.train)
            self.new_data = sigmoid_mapping.new_rating_mean_sigmoid_std_3(self.new_data)
            
            for i in self.new_data:
                for j in self.new_data[i]:
                    self.max_r_new.add(self.new_data[i][j])
            self.max_r_new = max(self.max_r_new)
            
        #  for new rating. 각 user의 평균을 사용하여 rating change. 4
        elif self.new == 4:
            self.new_data = copy.deepcopy(self.train)
            self.new_data = sigmoid_mapping.new_rating_mean_sigmoid_2_1(self.new_data)
            
            for i in self.new_data:
                for j in self.new_data[i]:
                    self.max_r_new.add(self.new_data[i][j])
            self.max_r_new = max(self.max_r_new)
            
        #  for new rating. 각 user의 평균과 분산을 사용하여 rating change. 5
        elif self.new == 5:
            self.new_data = copy.deepcopy(self.train)
            self.new_data = sigmoid_mapping.new_rating_mean_sigmoid_std_3_1(self.new_data)
            
            for i in self.new_data:
                for j in self.new_data[i]:
                    self.max_r_new.add(self.new_data[i][j])
            self.max_r_new = max(self.max_r_new)
    


        # return self.train, self.test

    # traditional similarity
    def trad_similarity(self):

        print('--------------------traditional {} similarity calculation--------------------'.format(self.measure))
        print()

        users = list(self.train.keys())
        n=1
        self.sim_d = {}

        for ui in tqdm(users):
            neighbors = users[n:]
            self.sim_d[ui] = {}
            
            if self.new == 0:  #  기본적인 data 로 유사도를 구함.
                for uj in neighbors:
                    if self.measure == 'cos':
                        self.sim_d[ui][uj] = similarity_methods.cosine_similarity(
                            self.train[ui],
                            self.train[uj])
                    if self.measure == 'pcc':
                        self.sim_d[ui][uj] = similarity_methods.PCC_similarity(
                            self.train[ui],
                            self.train[uj])
                    if self.measure == 'msd':
                        self.sim_d[ui][uj] = similarity_methods.MSD_similarity(
                            self.train[ui],
                            self.train[uj])
                    if self.measure == 'jac':
                        self.sim_d[ui][uj] = similarity_methods.Jaccard_similarity(
                            self.train[ui],
                            self.train[uj])
                    if self.measure == 'os':
                        self.sim_d[ui][uj] = similarity_methods.os(
                            self.train[ui],
                            self.train[uj],
                            self.N)
                    if self.measure == 'cos_jac':
                        cos = similarity_methods.cosine_similarity(
                            self.train[ui],
                            self.train[uj])
                        jac = similarity_methods.Jaccard_similarity(
                            self.train[ui],
                            self.train[uj]
                            )
                        self.sim_d[ui][uj] = (cos[0] * jac[0], cos[1])
                n+=1

            elif self.new != 0:  #  새로운 data 로 유사도를 구함.
                for uj in neighbors:
    
                    if self.measure == 'cos':
                        self.sim_d[ui][uj] = similarity_methods.cosine_similarity(
                            self.new_data[ui],
                            self.new_data[uj])
                    if self.measure == 'pcc':
                        self.sim_d[ui][uj] = similarity_methods.PCC_similarity(
                            self.new_data[ui],
                            self.new_data[uj])
                    if self.measure == 'msd':
                        self.sim_d[ui][uj] = similarity_methods.MSD_similarity(
                            self.new_data[ui],
                            self.new_data[uj])
                    if self.measure == 'jac':
                        self.sim_d[ui][uj] = similarity_methods.Jaccard_similarity(
                            self.new_data[ui],
                            self.new_data[uj])
                    if self.measure == 'os':
                        self.sim_d[ui][uj] = similarity_methods.os(
                            self.new_data[ui],
                            self.new_data[uj],
                            self.N)
                    if self.measure == 'os_new_rating':
                        self.sim_d[ui][uj] = similarity_methods.os_new_rating(
                            self.new_data[ui],
                            self.new_data[uj],
                            self.N)
                    if self.measure == 'os_new_rating_2times':
                        self.sim_d[ui][uj] = similarity_methods.os_new_rating_2times(
                            self.new_data[ui],
                            self.new_data[uj],
                            self.N)
                    if self.measure == 'cos_jac_new_rating2':
                        cos = similarity_methods.cosine_similarity(
                            self.new_data[ui],
                            self.new_data[uj])
                        jac = similarity_methods.Jaccard_similarity(
                            self.new_data[ui],
                            self.new_data[uj]
                            )
                        self.sim_d[ui][uj] = (cos[0] * jac[0], cos[1])
                n+=1

        users_r = users[::-1]
        n=1
        for ui in users_r:
            neighbor = users_r[n:]
            for uj in neighbor:
                self.sim_d[ui][uj] = self.sim_d[uj][ui]
            n+=1

        # return self.sim_d

    # proposed similarity
    def prop_similarity(self):

        print('--------------------proposed {} similarity calculation--------------------'.format(self.measure))
        print()

        users = list(self.train.keys())  # whole users
        self.sim_d = {}  # similarity matrics
        n=1

        sigmoid_dic_d = sigmoid_mapping.sigmoid_mapping_d(ratings_list=self.ratings_list,
                                                          mid=self.mid)
        sigmoid_dic_d_1 = sigmoid_mapping.sigmoid_mapping_d_1(ratings_list=self.ratings_list,
                                                              mid=self.mid)
        sigmoid_dic_d_2 = sigmoid_mapping.sigmoid_mapping_d_2(ratings_list=self.ratings_list,
                                                              mid=self.mid)
        sigmoid_dic_d_3 = sigmoid_mapping.sigmoid_mapping_d_3(ratings_list=self.ratings_list,
                                                              mid=self.mid)
        sigmoid_dic_d2 = sigmoid_mapping.sigmoid_mapping_d2(ratings_list=self.ratings_list,
                                                            mid=self.mid)
        sigmoid_dic_d2_1 = sigmoid_mapping.sigmoid_mapping_d2_1(ratings_list=self.ratings_list,
                                                                mid=self.mid)
        sigmoid_dic_d2_2 = sigmoid_mapping.sigmoid_mapping_d2_2(ratings_list=self.ratings_list,
                                                                mid=self.mid)
        sigmoid_dic_d3 = sigmoid_mapping.sigmoid_mapping_d3(ratings_list=self.ratings_list,
                                                            mid=self.mid)
        sigmoid_dic_d3_1 = sigmoid_mapping.sigmoid_mapping_d3_1(ratings_list=self.ratings_list,
                                                                mid=self.mid)
        sigmoid_dic_d3_2 = sigmoid_mapping.sigmoid_mapping_d3_2(ratings_list=self.ratings_list,
                                                                mid=self.mid)

        rd_pref = sigmoid_mapping.pref_ratio(self.train)

        if self.new != 0:  #  새로운 데이터를 사용함.
            for ui in tqdm(users):
                neighbors = users[n:]
                self.sim_d[ui] = {}

                for uj in neighbors:

                    if self.measure == 'os_sig':
                        self.sim_d[ui][uj] = sigmoid_mapping.os_sig_max(
                            ui=self.new_data[ui],
                            uj=self.new_data[uj],
                            N=self.N,
                            sigmoid_dic=sigmoid_dic_d_3,
                            max_r=self.max_r_new)
    
                    elif self.measure == 'os_sig_pos':
                        self.sim_d[ui][uj] = sigmoid_mapping.os_sig_max(
                            ui=self.new_data[ui],
                            uj=self.new_data[uj],
                            N=self.N,
                            sigmoid_dic=sigmoid_dic_d3_2,
                            max_r=self.max_r_new)
    
                    elif self.measure == 'os_sig_neg':
                        self.sim_d[ui][uj] = sigmoid_mapping.os_sig_max(
                            ui=self.new_data[ui],
                            uj=self.new_data[uj],
                            N=self.N,
                            sigmoid_dic=sigmoid_dic_d2_2,
                            max_r=self.max_r_new)
                n+=1
            
        elif self.new == 0:  # 기본 데이터를 사용함.
            for ui in tqdm(users):
                neighbors = users[n:]
                self.sim_d[ui] = {}
    
                for uj in neighbors:
    
                    # compare paper and propose
                    if self.measure == 'os_sig1':
                        self.sim_d[ui][uj] = sigmoid_mapping.os_sig_no_euclidean(
                            ui=self.train[ui],
                            uj=self.train[uj],
                            N=self.N,
                            sigmoid_dic=sigmoid_dic_d_3,
                            max_r=self.max_r)
                        
                    if self.measure == 'os_sig2':
                        self.sim_d[ui][uj] = sigmoid_mapping.os_sig_no_euclidean_no_max(
                            ui=self.train[ui],
                            uj=self.train[uj],
                            N=self.N,
                            sigmoid_dic=sigmoid_dic_d_3,
                            max_r=self.max_r)
    
                    elif self.measure == 'os_sig_pos':
                        self.sim_d[ui][uj] = sigmoid_mapping.os_sig_no_euclidean_no_max(
                            ui=self.train[ui],
                            uj=self.train[uj],
                            N=self.N,
                            sigmoid_dic=sigmoid_dic_d3_2,
                            max_r=self.max_r)
    
                    elif self.measure == 'os_sig_neg':
                        self.sim_d[ui][uj] = sigmoid_mapping.os_sig_no_euclidean_no_max(
                            ui=self.train[ui],
                            uj=self.train[uj],
                            N=self.N,
                            sigmoid_dic=sigmoid_dic_d2_2,
                            max_r=self.max_r)

    ################################################################
                    elif self.measure == 'sig1':
                        self.sim_d[ui][uj] = sigmoid_mapping.sigmoid_mapping_similarity(
                            ui=self.train[ui],
                            uj=self.train[uj],
                            N=self.N,
                            sigmoid_dic=sigmoid_dic_d_3)  # sqrt(mid)
    
                    elif self.measure == 'sig':
                        self.sim_d[ui][uj] = sigmoid_mapping.sigmoid_mapping_similarity(
                            self.train[ui],
                            self.train[uj],
                            sigmoid_dic=sigmoid_dic_d)
                    elif self.measure == 'sig_jac':
                        sig = sigmoid_mapping.sigmoid_mapping_similarity(
                            self.train[ui],
                            self.train[uj],
                            sigmoid_dic=sigmoid_dic_d)
                        jac = similarity_methods.Jaccard_similarity(
                            self.train[ui],
                            self.train[uj])
                        self.sim_d[ui][uj] = (sig[0] * jac[0], sig[1])
    
                    elif self.measure == 'sig2':
                        self.sim_d[ui][uj] = sigmoid_mapping.sigmoid_mapping_similarity(
                            self.train[ui],
                            self.train[uj],
                            self.N,
                            sigmoid_dic=sigmoid_dic_d_3)
                    elif self.measure == 'sig2_jac':
                        sig = sigmoid_mapping.sigmoid_mapping_similarity(
                            self.train[ui],
                            self.train[uj],
                            sigmoid_dic=sigmoid_dic_d2)
                        jac = similarity_methods.Jaccard_similarity(
                            self.train[ui],
                            self.train[uj])
                        self.sim_d[ui][uj] = (sig[0] * jac[0], sig[1])
    
                    elif self.measure == 'sig2_1':
                        self.sim_d[ui][uj] = sigmoid_mapping.sigmoid_mapping_similarity(
                            self.train[ui],
                            self.train[uj],
                            sigmoid_dic=sigmoid_dic_d2_1)
                    elif self.measure == 'sig2_1_jac':
                        sig = sigmoid_mapping.sigmoid_mapping_similarity(
                            self.train[ui],
                            self.train[uj],
                            sigmoid_dic=sigmoid_dic_d2_1)
                        jac = similarity_methods.Jaccard_similarity(
                            self.train[ui],
                            self.train[uj])
                        self.sim_d[ui][uj] = (sig[0] * jac[0], sig[1])
    
                    elif self.measure == 'sig3':
                        sig = sigmoid_mapping.sigmoid_mapping_similarity(
                            self.train[ui],
                            self.train[uj],
                            self.N,
                            sigmoid_dic=sigmoid_dic_d_3)
                        pf = 1 - abs(rd_pref[ui]['exp'] - rd_pref[uj]['exp'])
                        self.sim_d[ui][uj] = (sig[0] * pf, sig[1])
                    elif self.measure == 'sig3_jac':
                        sig = sigmoid_mapping.sigmoid_mapping_similarity(
                            self.train[ui],
                            self.train[uj],
                            sigmoid_dic=sigmoid_dic_d)
                        jac = similarity_methods.Jaccard_similarity(
                            self.train[ui],
                            self.train[uj])
                        pf = 1 - abs(rd_pref[ui]['exp'] - rd_pref[uj]['exp'])
                        self.sim_d[ui][uj] = (sig[0] * jac[0] * pf, sig[1])
    
                    elif self.measure == 'sig4':
                        self.sim_d[ui][uj] = sigmoid_mapping.sigmoid_mapping_similarity(
                            self.train[ui],
                            self.train[uj],
                            sigmoid_dic=sigmoid_dic_d3)
                    elif self.measure == 'sig4_jac':
                        sig = sigmoid_mapping.sigmoid_mapping_similarity(
                            self.train[ui],
                            self.train[uj],
                            sigmoid_dic=sigmoid_dic_d3)
                        jac = similarity_methods.Jaccard_similarity(
                            self.train[ui],
                            self.train[uj])
                        self.sim_d[ui][uj] = (sig[0] * jac[0], sig[1])
                        
                n+=1

        users_r = users[::-1]
        n=1
        for ui in users_r:
            neighbor = users_r[n:]
            for uj in neighbor:
                self.sim_d[ui][uj] = self.sim_d[uj][ui]
            n+=1
        # return self.sim_d

    # predict using knn with mean.
    def predict_kNN_Mean(self):

        print('-------------------- predict knn mean--------------------')
        print()

        self.predict_d = {}  # result
        users = list(self.train.keys())  # whole users
        items = []  # whole items
        for i in self.data:
            items.extend(list(self.data[i].keys()))
        items = set(items)

        users_avg = {i: np.mean(list(self.train[i].values())) for i in self.train}

        for ui in tqdm(users):
            unrated = items.difference(set(self.train[ui].keys()))  # unrated items
            k_neighbor = sorted(self.sim_d[ui].items(),
                                key=lambda x: x[1],
                                reverse=True)[:self.k]
            self.predict_d[ui] = {}

            for un in unrated:
                self.predict_d[ui][un] = users_avg[ui]
                up = []
                down = []

                for uj, sim in k_neighbor:
                    if un in self.train[uj].keys():
                        up.append(sim[0] * (self.train[uj][un] - users_avg[uj]))
                        down.append(sim[0])
                try:
                    self.predict_d[ui][un] += round(sum(up), 5) / round(sum(down), 5)
                except ZeroDivisionError:
                    pass
                if math.isnan(self.predict_d[ui][un]) or math.isinf(self.predict_d[ui][un]):
                    self.predict_d[ui][un] = users_avg[ui]

        # return self.predict_d

    # predict using knn with basic
    def predict_kNN_Basic(self):

        print('-------------------- predict knn basic--------------------')
        print()

        self.predict_d = {}  # result
        users = list(self.train.keys())  # whole users
        items = []  # whole items
        for i in self.data:
            items.extend(list(self.data[i].keys()))
        items = set(items)

        users_avg = {i: np.mean(list(self.train[i].values())) for i in self.train}

        for ui in tqdm(users):
            unrated = items.difference(set(self.train[ui].keys()))  # unrated items
            k_neighbor = sorted(self.sim_d[ui].items(),
                                key=lambda x: x[1],
                                reverse=True)[:self.k]
            self.predict_d[ui] = {}

            for un in unrated:
                self.predict_d[ui][un] = 0
                up = []
                down = []

                for uj, sim in k_neighbor:
                    if un in self.train[uj].keys():
                        up.append(sim[0] * (self.train[uj][un]))
                        down.append(sim[0])
                try:
                    self.predict_d[ui][un] = sum(up) / sum(down)
                    if np.isnan(self.predict_d[ui][un]):
                        self.predict_d[ui][un] = users_avg[ui]
                except ZeroDivisionError:
                    self.predict_d[ui][un] = users_avg[ui]

            if self.predict_d[ui][un] == 0:
                self.predict_d[ui][un] = users_avg[ui]

        # return self.predict_d

    # MAE performance calculation
    def performance_mae(self):

        print('-------------------- performance mae calculation--------------------')
        print()

        users = self.data.keys()  # whole users

        self.mae = 0
        self.pred_ = []
        self.real_ = []

        for i in users:
            for j in self.test[i]:
                self.pred_.append(self.predict_d[i][j])
                self.real_.append(self.test[i][j])

        self.result = [abs(p - r) for p, r in zip(self.pred_, self.real_)]
        self.mae = sum(self.result) / len(self.result)

    # baseline performance calculation
    def performance_mae_baseline(self):

        print('-------------------- performance mae baseline--------------------')
        print()

        users = self.test.keys()  # whole users
        users_avg = {i: np.mean(list(self.data[i].values())) for i in self.data}

        self.performance_mae_baseline = 0
        self.pred_ = []
        self.real_ = []

        for i in users:
            for j in self.test[i]:
                self.pred_.append(users_avg[i])
                self.real_.append(self.test[i][j])

        self.result = [abs(p - r) for p, r in zip(self.pred_, self.real_)]
        self.performance_mae_baseline = sum(self.result) / len(self.result)

    #  traditional similarity
    def run_e1(self): 
        self.train_test_split()
        self.trad_similarity()
        self.predict_kNN_Mean()
        self.performance_mae()

        return self.mae

    #  proposed similarity
    def run_e2(self):
        self.train_test_split()
        self.prop_similarity()
        self.predict_kNN_Mean()
        self.performance_mae()

        return self.mae



#%%

rd = load_data.user_item_dictionary()
rd_1m = load_data.user_item_dictionary_1M()

def experiment(data, test_ratio, cv, measure, k, soso, new):
    d = {}
    for sim in measure:
        d[sim] = {}
        for k_ in k:
            d[sim][k_]={}

            cf = CF(data=data, test_ratio=test_ratio, CV=cv, measure=sim, k=k_, soso=soso, new=new)

            for cv_ in range(cv):

                print('-----------------sim = {}  k = {}  cv = {}-------------------------------'.format(sim,k_,cv_+1))
                
                if sim in ['cos', 'pcc', 'msd', 'jac', 'os', 'os_new_rating', 'os_new_rating_2times', 'cos_jac_new_rating2', 'cos_jac']:
                    cf.run_e1()
                    d[sim][k_][cv_] = cf.mae
                else:
                    cf.run_e2()
                    d[sim][k_][cv_] = cf.mae


    agg_d = {}   
    for sim in measure:
        agg_d[sim] = {}
        for k_ in k:
            agg_d[sim][k_] = 0
            basket = []
            for cv_ in range(cv):
                basket.append(d[sim][k_][cv_])
            agg_d[sim][k_] = sum(basket) / len(basket)
    return d, agg_d

# 5-fold-CV

# 실험 1.
e_os, e_os_agg = experiment(data=rd, test_ratio=0.2, cv=5, measure=['os'], k=list(range(10,101,10)), soso=3, new=2)
e_os1, e_os_agg1 = experiment(data=rd, test_ratio=0.2, cv=5, measure=['os_sig1'], k=list(range(10,101,10)), soso=3, new=0)
e_os2, e_os_agg2 = experiment(data=rd, test_ratio=0.2, cv=5, measure=['os_sig2'], k=list(range(10,101,10)), soso=3, new=0)

# 실험 1. 1M data set
e_os, e_os_agg = experiment(data=rd_1m, test_ratio=0.2, cv=5, measure=['os'], k=[10], soso=3, new=2)
e_os1, e_os_agg1 = experiment(data=rd_1m, test_ratio=0.2, cv=5, measure=['os_sig1'], k=list(range(10,101,10)), soso=3, new=0)
e_os2, e_os_agg2 = experiment(data=rd_1m, test_ratio=0.2, cv=5, measure=['os_sig2'], k=list(range(10,101,10)), soso=3, new=0)


# 실험 2.
st, st_agg = experiment(data=rd, test_ratio=0.2, cv=5, measure=['cos','pcc','msd','os'], k=list(range(10,101,10)), soso=3, new=0)
st1, st_agg1 = experiment(data=rd, test_ratio=0.2, cv=5, measure=['cos','pcc','msd','os_new_rating','os_new_rating_2times'], k=list(range(10,101,10)), soso=3, new=1)
st1_1, st_agg1_1 = experiment(data=rd, test_ratio=0.2, cv=5, measure=['cos','pcc','msd','os_new_rating','os_new_rating_2times'], k=list(range(10,101,10)), soso=3, new=1.5)
st2, st_agg2 = experiment(data=rd, test_ratio=0.2, cv=5, measure=['cos','pcc','msd','os_new_rating','os_new_rating_2times'], k=list(range(10,101,10)), soso=3, new=2)
st3, st_agg3 = experiment(data=rd, test_ratio=0.2, cv=5, measure=['cos','pcc','msd','os_new_rating','os_new_rating_2times'], k=list(range(10,101,10)), soso=3, new=3)
st2_1, st_agg2_1 = experiment(data=rd, test_ratio=0.2, cv=5, measure=['cos','pcc','msd','os_new_rating','os_new_rating_2times'], k=list(range(10,101,10)), soso=3, new=4)
st3_1, st_agg3_1 = experiment(data=rd, test_ratio=0.2, cv=5, measure=['cos','pcc','msd','os_new_rating','os_new_rating_2times'], k=list(range(10,101,10)), soso=3, new=5)

# 실험 2. 1M data set
# 기존 유사도 성능.
st_cos, st_agg_cos = experiment(data=rd_1m, test_ratio=0.2, cv=5, measure=['cos'], k=list(range(10,101,10)), soso=3, new=0)
st_pcc, st_agg_pcc = experiment(data=rd_1m, test_ratio=0.2, cv=5, measure=['pcc'], k=list(range(10,101,10)), soso=3, new=0)
st_msd, st_agg_msd = experiment(data=rd_1m, test_ratio=0.2, cv=5, measure=['msd'], k=list(range(10,101,10)), soso=3, new=0)
st_os, st_agg_os = experiment(data=rd_1m, test_ratio=0.2, cv=5, measure=['os'], k=list(range(10,101,10)), soso=3, new=0)

# new_rating2 사용한 기존 유사도 성능.
st_cos_n2, st_agg_cos_n2 = experiment(data=rd_1m, test_ratio=0.2, cv=5, measure=['cos'], k=list(range(10,101,10)), soso=3, new=2)
st_pcc_n2, st_agg_pcc_n2 = experiment(data=rd_1m, test_ratio=0.2, cv=5, measure=['pcc'], k=list(range(10,101,10)), soso=3, new=2)
st_msd_n2, st_agg_msd_n2 = experiment(data=rd_1m, test_ratio=0.2, cv=5, measure=['msd'], k=list(range(10,101,10)), soso=3, new=2)
st_os_n2, st_agg_os_n2 = experiment(data=rd_1m, test_ratio=0.2, cv=5, measure=['os_new_rating'], k=list(range(10,101,10)), soso=3, new=2)


st1, st_agg1 = experiment(data=rd_1m, test_ratio=0.2, cv=5, measure=['cos','pcc','msd','os_new_rating','os_new_rating_2times'], k=list(range(10,101,10)), soso=3, new=1)
st1_1, st_agg1_1 = experiment(data=rd_1m, test_ratio=0.2, cv=5, measure=['cos','pcc','msd','os_new_rating','os_new_rating_2times'], k=list(range(10,101,10)), soso=3, new=1.5)
st2, st_agg2 = experiment(data=rd_1m, test_ratio=0.2, cv=5, measure=['cos','pcc','msd','os_new_rating','os_new_rating_2times'], k=list(range(10,101,10)), soso=3, new=2)
st3, st_agg3 = experiment(data=rd_1m, test_ratio=0.2, cv=5, measure=['cos','pcc','msd','os_new_rating','os_new_rating_2times'], k=list(range(10,101,10)), soso=3, new=3)
st4, st_agg4 = experiment(data=rd_1m, test_ratio=0.2, cv=5, measure=['cos','pcc','msd','os_new_rating','os_new_rating_2times'], k=list(range(10,101,10)), soso=3, new=4)
st5, st_agg5 = experiment(data=rd_1m, test_ratio=0.2, cv=5, measure=['cos','pcc','msd','os_new_rating','os_new_rating_2times'], k=list(range(10,101,10)), soso=3, new=5)


# jaccard similarity.
jac, jac_agg = experiment(data=rd_1m, test_ratio=0.2, cv=5, measure=['jac'], k=[10], soso=3, new=0)

# 추가실험 cosine * jaccard
# cosine, jaccard using base rating.
cj, cj_agg = experiment(data=rd, test_ratio=0.2, cv=5, measure=['cos_jac'], k=list(range(10,101,10)), soso=3, new=0)

# cosine, jaccard using new rating2.
cj, cj_agg = experiment(data=rd, test_ratio=0.2, cv=5, measure=['cos_jac_new_rating2'], k=list(range(10,101,10)), soso=3, new=2)
#%%
# visualization
for i in ['os','os_sig1','os_sig2']:
    plt.plot(list(agg_d[i].keys()), list(agg_d[i].values()),
             ls='--',
             marker='.',
             markersize='7',
             label=i)
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.title('5-fold validation')
plt.ylabel('MAE')
plt.xlabel('k neighbors')


#%%


# save result
with open('result/result_{}_pcc_1M.pickle'.format(str(datetime.datetime.now())[:13]+'시'+str(datetime.datetime.now())[14:16]+'분'), 'wb') as f:
    pickle.dump(st_agg_pcc, f)

# load result
with open('result/result_rating3_2020-12-13 21시35분.pickle', 'rb') as f:
    past_result3 = pickle.load(f)


def combine_result(past, new):
    sims = set(new.keys()).difference(set(past.keys()))
    for sim in sims:
        past[sim] = {}
        for k in new[sim]:
            past[sim][k] = new[sim][k]

    return past

agg_d = combine_result(agg_d, past_result3)

    