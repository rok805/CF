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

    def __init__(self, data, test_ratio, random_state, measure, k, soso=3, new=0):
        self.data = data
        self.new_data = None
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.measure = measure
        self.k = k
        self.mid = soso
        self.new = new

        #  for rating sigmoid mapping.
        self.ratings_list = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
        
        self.max_r = max(self.ratings_list)

        #  for new_rating_mean method
        self.max_r_new = set()


    # train/test split
    def train_test_split(self):

        print('-----train test split-----')
        print()

        self.train = {}
        self.test = {}

        for user in self.data:

            item_rate = list(self.data[user].items())
            random.Random(self.random_state).shuffle(item_rate)
            length = len(item_rate)
            cri = int(length * (1-0.2))

            self.train[user] = {i: r for i, r in item_rate[:cri]}
            self.test[user] = {i: r for i, r in item_rate[cri:]}

        # the number of ratings in train set.
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


        #  for new rating. 각 user의 평균을 사용하여 rating change.
        if self.new == 1:
            self.new_data = copy.deepcopy(self.train)
            self.new_data = sigmoid_mapping.new_rating_mean_1(self.new_data)  # new rating method
    
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
        
        #  for new rating. 각 user의 평균과 분산을 사용하여 rating change.
        elif self.new == 3:
            self.new_data = copy.deepcopy(self.train)
            self.new_data = sigmoid_mapping.new_rating_mean_sigmoid_std_3(self.new_data)  # new rating method 2 번째 방법.
            
            for i in self.new_data:
                for j in self.new_data[i]:
                    self.max_r_new.add(self.new_data[i][j])
            self.max_r_new = max(self.max_r_new)
    


        # return self.train, self.test

    # traditional similarity
    def trad_similarity(self):

        print('-----traditional {} similarity calculation-----'.format(self.measure))
        print()

        users = self.train.keys()
        self.sim_d = {}


        for ui in tqdm(users):
            neighbors = list(users)
            neighbors.remove(ui)

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

        # return self.sim_d

    # proposed similarity
    def prop_similarity(self):

        print('-----proposed {} similarity calculation-----'.format(self.measure))
        print()

        users = self.train.keys()  # whole users
        self.sim_d = {}  # similarity matrics

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
                neighbors = list(users)
                neighbors.remove(ui)

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
            
        elif self.new == 0:  # 기본 데이터를 사용함.
            for ui in tqdm(users):
                neighbors = list(users)
                neighbors.remove(ui)
    
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
        # return self.sim_d

    # predict using knn with mean.
    def predict_kNN_Mean(self):

        print('----- predict knn mean-----')
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

        print('----- predict knn basic-----')
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

        print('----- performance mae calculation-----')
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

        print('----- performance mae baseline-----')
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

def experiment(data, test_ratio, random_state, measure, k, soso, new):
    d = {}
    for sim in measure:
        d[sim] = {}
        for k_ in k:
            d[sim][k_]={}
            for rs in random_state:
                cf = CF(data=data, test_ratio=test_ratio, random_state=rs, measure=sim, k=k_, soso=soso, new=new)
                
                print('-----------------sim = {}  k = {}  rs = {}-------------------------------'.format(sim,k_,rs))
                
                if sim in ['cos', 'pcc', 'msd', 'jac', 'os', 'os_new_rating', 'os_new_rating_2times']:
                    cf.run_e1()
                    d[sim][k_][rs] = cf.mae
                else:
                    cf.run_e2()
                    d[sim][k_][rs] = cf.mae


    agg_d = {}   
    for sim in measure:
        agg_d[sim] = {}
        for k_ in k:
            agg_d[sim][k_] = 0
            basket = []
            for rs in random_state:
                basket.append(d[sim][k_][rs])
            agg_d[sim][k_] = sum(basket) / len(basket)
    return d, agg_d


st1, st_agg1 = experiment(data=rd, test_ratio=0.2, random_state=[1,2,3,4,5], measure=['os_new_rating','os_new_rating_2times'], k=list(range(10,101,10)), soso=3, new=1)
st2, st_agg2 = experiment(data=rd, test_ratio=0.2, random_state=[1,2,3,4,5], measure=['os_new_rating','os_new_rating_2times'], k=list(range(10,101,10)), soso=3, new=2)
st3, st_agg3 = experiment(data=rd, test_ratio=0.2, random_state=[1,2,3,4,5], measure=['cos','pcc','msd','os_new_rating','os_new_rating_2times'], k=list(range(10,101,10)), soso=3, new=3)


st1, st_agg1 = experiment(data=rd, test_ratio=0.2, random_state=[1], measure=['os_new_rating','os_new_rating_2times'], k=[10], soso=3, new=1)
st2, st_agg2 = experiment(data=rd, test_ratio=0.2, random_state=[1], measure=['os_new_rating','os_new_rating_2times'], k=[10], soso=3, new=2)
st3, st_agg3 = experiment(data=rd, test_ratio=0.2, random_state=[1], measure=['cos','pcc','msd','os_new_rating','os_new_rating_2times'], k=[10], soso=3, new=3)

#%%
# visualization
for i in ['cos','pcc','msd','os']:
    plt.plot(list(past_result3[i].keys()), list(past_result3[i].values()),
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
with open('result/result_{}.pickle'.format(str(datetime.datetime.now())[:13]+'시'+str(datetime.datetime.now())[14:16]+'분'), 'wb') as f:
    pickle.dump(agg_d, f)

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

agg_d = combine_result(agg_d, past_result2)
    
    
    