# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 21:48:14 2020

@author: user
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
            self.new_data = sigmoid_mapping.new_rating_mean_sigmoid_std_3(self.new_data)  # new rating method 2 번째 방법.
            
            for i in self.new_data:
                for j in self.new_data[i]:
                    self.max_r_new.add(self.new_data[i][j])
            self.max_r_new = max(self.max_r_new)
            
        #  for new rating. 각 user의 평균을 사용하여 rating change. 4
        elif self.new == 4:
            self.new_data = copy.deepcopy(self.train)
            self.new_data = sigmoid_mapping.new_rating_mean_sigmoid_2_1(self.new_data)  # new rating method 2 번째 방법.
            
            for i in self.new_data:
                for j in self.new_data[i]:
                    self.max_r_new.add(self.new_data[i][j])
            self.max_r_new = max(self.max_r_new)
            
        #  for new rating. 각 user의 평균과 분산을 사용하여 rating change. 5
        elif self.new == 5:
            self.new_data = copy.deepcopy(self.train)
            self.new_data = sigmoid_mapping.new_rating_mean_sigmoid_std_3_1(self.new_data)  # new rating method 2 번째 방법.
            
            for i in self.new_data:
                for j in self.new_data[i]:
                    self.max_r_new.add(self.new_data[i][j])
            self.max_r_new = max(self.max_r_new)


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


        # return self.train, self.test
rd = load_data.user_item_dictionary()      
users = list(rd.keys())
n=1
sim_d={}
for ui in users:
    sim_d[ui]={}
    neighbor=users[n:]
    for uj in neighbor:
        sim_d[ui][uj] = similarity_methods.cosine_similarity(
            rd[ui],
            rd[uj]
            )
    n+=1

users_r = users[::-1]
n=1
for ui in users_r:
    neighbor=users_r[n:]
    for uj in neighbor:
        sim_d[ui][uj]=sim_d[uj][ui]
    n+=1
# traditional similarity
    def trad_similarity(self):

        print('--------------------traditional {} similarity calculation--------------------'.format(self.measure))
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
                        self.sim_d[ui][uj] = sim_d[uj][ui]
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


        # return self.sim_d
        
