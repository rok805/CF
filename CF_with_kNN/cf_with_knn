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
import random
import pickle
import math


class CF:

    def __init__(self, data, test_ratio, random_state, measure, k, soso=3):
        self.data = data
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.measure = measure
        self.k = k
        self.mid = soso

        # new rating mapping.
        self.ratings_list = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

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

        # return self.sim_d

    # proposed similarity
    def prop_similarity(self):

        print('-----proposed {} similarity calculation-----'.format(self.measure))
        print()

        users = self.train.keys()
        self.sim_d = {}

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

        for ui in tqdm(users):
            neighbors = list(users)
            neighbors.remove(ui)

            self.sim_d[ui] = {}

            for uj in neighbors:

                # compare paper and propose
                if self.measure == 'os_sig':
                    self.sim_d[ui][uj] = sigmoid_mapping.os_sig_max(
                        ui=self.train[ui],
                        uj=self.train[uj],
                        N=self.N,
                        sigmoid_dic=sigmoid_dic_d_3)

                elif self.measure == 'os_sig_pos':
                    self.sim_d[ui][uj] = sigmoid_mapping.os_sig_max(
                        ui=self.train[ui],
                        uj=self.train[uj],
                        N=self.N,
                        sigmoid_dic=sigmoid_dic_d2_1)

                elif self.measure == 'os_sig_neg':
                    self.sim_d[ui][uj] = sigmoid_mapping.os_sig_max(
                        ui=self.train[ui],
                        uj=self.train[uj],
                        N=self.N,
                        sigmoid_dic=sigmoid_dic_d3_1)

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
                        sigmoid_dic=sigmoid_dic_d2_2)
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

        self.performance_mae = 0
        self.pred_ = []
        self.real_ = []

        for i in users:
            for j in self.test[i]:
                self.pred_.append(self.predict_d[i][j])
                self.real_.append(self.test[i][j])

        self.result = [abs(p - r) for p, r in zip(self.pred_, self.real_)]
        self.performance_mae = sum(self.result) / len(self.result)

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

        return self.performance_mae

    #  proposed similarity
    def run_e2(self):
        self.train_test_split()
        self.prop_similarity()
        self.predict_kNN_Mean()
        self.performance_mae()

        return self.performance_mae

# %%
# x = CF(data=rd, test_ratio=0.2, random_state=1, measure='os_sig', k=5)

        
rd = load_data.user_item_dictionary()
similarity_measures = ['sig1','sig2','sig3'] # 'os', 'os_sig', 'os_sig_pos', 'os_sig_neg'
random_states = [1, 2, 3, 4, 5]
k_ = list(range(10, 101, 10))

experiment_result = {}
for k in k_:
    experiment_result[k] = {}

    for rs in random_states:
        experiment_result[k][rs] = {}

        for mea in similarity_measures:
            print()
            print('==== k: {} ==== random_state: {} ==== measure: {} ===='.format(k, rs, mea))
            print()
            cf = CF(data=rd, test_ratio=0.2, random_state=rs, measure=mea, k=k)

            if mea in ['cos', 'pcc', 'msd', 'jac', 'os']:
                cf.run_e1()
                experiment_result[k][rs][mea] = cf.performance_mae
            else:
                cf.run_e2()
                experiment_result[k][rs][mea] = cf.performance_mae


        # baseline code.
        # cf.performance_mae_baseline()
        # experiment_result[k][rs]['baseline'] = cf.performance_mae_baseline



rd = load_data.user_item_dictionary()
similarity_measures2 = ['os_sig']
random_states2 = [1, 2, 3, 4, 5]
k_2 = list(range(10, 101, 10))

experiment_result2 = {}
for k in k_2:
    experiment_result2[k] = {}

    for rs in random_states2:
        experiment_result2[k][rs] = {}

        for mea in similarity_measures2:
            print()
            print(f'==== k: {k} ==== random_state: {rs} ==== measure: {mea} ====')
            print()
            cf = CF(data=rd, test_ratio=0.2, random_state=rs, measure=mea, k=k, soso='mean')

            if mea in ['cos', 'pcc', 'msd', 'jac', 'os']:
                cf.run_e1()
                experiment_result2[k][rs][mea] = cf.performance_mae
            else:
                cf.run_e2()
                experiment_result2[k][rs][mea] = cf.performance_mae

# %%

# k-fold aggregation
agg_d = {}
s_m = similarity_measures
for m in s_m:
    agg_d[m] = {}
    for i in k_:
        agg_d[m][i] = 0
        basket = []
        for j in random_states:
            basket.append(experiment_result[i][j][m])
        agg_d[m][i] = sum(basket) / len(random_states)


# visualization
for i in agg_d:
    plt.plot(list(agg_d[i].keys()), list(agg_d[i].values()),
             ls='--',
             marker='.',
             markersize='7',
             label=i)
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.title('5-fold validation, mid=3')
plt.ylabel('MAE')
plt.xlabel('k neighbors')

# write result
with open('e_os1130_mine_d_3(9).pickle', 'wb') as f:
    pickle.dump(agg_d, f)

# save result
with open('e_os1127_desc_mid3_k_100(4).pickle', 'rb') as f:
    agg = pickle.load(f)

x={}
# combine with pre result
for m in similarity_measures:
    x[m] = {}
    for i in [10,20,30,40,50,60,70,80,90,100]:
        x[m][i] = agg_d[m][i]
        agg_d=x
agg_d = agg

