# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 17:09:52 2020

@author: user
"""


#%% class 

from scipy.sparse import csr_matrix, csc_matrix
from numpy.linalg import norm
from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime
import random
import pickle
import math

from load_data import data_load


# 데이터 로드하기.

data = data_load.create_rating()
# data_1m = data_load.create_rating_1M()

data_d = data_load.create_rating_dic()
# data_1m_d = data_load.create_rating_dic_1M()

##############################################################################

class CFwithKnn:
    
    def __init__(self, data, data_d, k, CV, sim):
        self.data = data
        self.data_d = data_d
        self.k = k
        self.CV = CV
        self.cv = 0
        self.sim = sim
        self.mid=3
        
        self.rating_list = set(self.data['rating']) # unique ratings.

    # 1. train/test set 분리하기.
    # test set 으로 분리될 평점.
    def cv_div(self, x):
        idx = list(range(len(x)))
        div = len(x) // self.CV + 1
        div_mok = len(idx) // self.CV
        can = []
    
        if div-div_mok > 0:
            over = len(idx) - div_mok * self.CV
    
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


    def train_test_split(self):
        print()
        print('========================== split data set =================================')
        print('=========================cv: {}=========================='.format(self.cv+1))

        self.train_d = {}
        self.test_d = {}
    
        for user in self.data_d:
    
            item_ = list(self.data_d[user].items())
            random.Random(7777).shuffle(item_)
            length = len(item_)
    
            basket = self.cv_div(item_)[self.cv]
    
            self.train_d[user] = {item_[i][0]:item_[i][1] for i in range(length) if i not in basket}
            self.test_d[user] = {item_[i][0]:item_[i][1] for i in range(length) if i in basket}
    
        
        # user 별 평균 점수 딕셔너리 생성.
        self.data_mean = {}
        for i in self.train_d:
            self.data_mean[i] = np.mean(list(self.train_d[i].values()))
            
        self.cv+=1



##############################################################################


    # matrix 생성을 위한 index 추출하기.
    def make_matrix(self):
        users=[]
        items=[]
        ratings=[]
        
        for user in self.train_d:
            for item in self.train_d[user]:
                users.append(user)
                items.append(item)
                ratings.append(self.data_d[user][item])
    
        # row, column index를 0부터 시작하게 함.    
        users = np.array(users)
        items = np.array(items)
    
        self.data_csr_matrix = csr_matrix((ratings, (users, items)))
        self.data_matrix = self.data_csr_matrix.toarray()
    
        



##############################################################################
    
    # 실험 1을 위한 rating mapping.
    
    
    def sigmoid_mapping(self):
    
        self.sigmoid_dic = {}
    
        for i in self.rating_list:
            self.sigmoid_dic[i] = 1 / (1 + np.exp(-i + self.mid))


##############################################################################

    
    # 유사도 지표.
    def cosine(self, ui, uj):
        up = np.dot(ui,uj)
        down = norm(ui)*norm(uj)
        
        try:
            return up/down
        except:
            return 0
    
    def pearson_correlation(self, ui, uj, mi, mj):
        if len(ui) < 2:
            return 0
        up = np.multiply((ui-mi),(uj-mj)).sum()
        down = norm((ui-mi))*norm((uj-mj))
        
        try:
            s = up / down
            if not math.isnan(s):
                return s
            else:
                return 0
        except:
            return 0
        
    def mean_squared_difference(self, ui, uj):
        if len(ui) < 2:
            return 0
        up = ((ui-uj)**2).sum()
        down = len(ui)
        
        try:
            s = 1 - up/down
            if not math.isnan(s):
                return s
            else:
                return 0
        except:
            return 0
    
    def pairwise_max(self, a, b):
        if a > b:
            return a
        else:
            return b  
        
    def os(self, ui, uj):
        #PNCR
        pncr = np.exp(-(self.item_length-len(ui))/self.item_length)
    
        #ADF
        vfunc = np.vectorize(self.pairwise_max)
        adf = (np.exp(-abs(ui-uj)/vfunc(ui,uj))).sum() / len(ui)
        try:
            return pncr * adf
        except:
            return pncr * adf
    
    def os_sig(self, ui, uj):
        #PNCR
        pncr = np.exp(-(self.item_length-len(ui))/self.item_length)
        ui2 = np.array([self.sigmoid_dic[i] for i in ui])
        uj2 = np.array([self.sigmoid_dic[j] for j in uj])
    
        #ADF
        adf = (np.exp(-abs(ui2-uj2)/max(self.rating_list))).sum() / len(ui2)
    
        try:
            return pncr * adf
        except:
            return pncr * adf


##############################################################################
    # 유사도 행렬 만들기.
    
    def similarity_calculation(self):
        print()
        print('========================== similarity =================================')
        print('========================== similarity:{}  k:{}========================='.format(self.sim, self.k))


        self.user_length = len(self.train_d)               # 총 user 수.
        self.item_length = len(set(self.data['movieId'])) # 총 item 수.
        users = list(range(self.user_length))        # user ID.
        n = 1
        
        self.sim_mat = np.zeros((self.user_length, self.user_length), dtype=float) # 유사도 행렬.
        
        for user in tqdm(users):
            neighbor = users[n:]
            n+=1
            for nei in neighbor:
                co_item = np.array(list(set(self.train_d[user].keys()).intersection(set(self.train_d[nei].keys()))))
        
                if self.sim == 'cos':
                    try:
                        self.sim_mat[user][nei] = self.cosine(ui=self.data_matrix[user,co_item],
                                                    uj=self.data_matrix[nei,co_item])
                    except IndexError:
                        self.sim_mat[user][nei] = 0
        
                elif self.sim == 'pcc':
                    try:
                        self.sim_mat[user][nei] = self.pearson_correlation(ui=self.data_matrix[user,co_item],
                                                                           uj=self.data_matrix[nei,co_item],
                                                                           mi=self.data_mean[user],
                                                                           mj=self.data_mean[nei])
                    except IndexError:
                        self.sim_mat[user][nei] = 0
                
                elif self.sim == 'msd':
                    try:
                        self.sim_mat[user][nei] = self.mean_squared_difference(ui=self.data_matrix[user,co_item],
                                                                               uj=self.data_matrix[nei,co_item])
                    except IndexError:
                        self.sim_mat[user][nei] = 0
                        
                elif self.sim == 'jac':
                    try:
                        self.sim_mat[user][nei] = len(co_item) / len(set(self.train_d[user].keys()).union(set(self.train_d[nei].keys())))
                    except IndexError:
                        self.sim_mat[user][nei] = 0
        
                elif self.sim == 'os':
                    try:
                        self.sim_mat[user][nei] = self.os(ui=self.data_matrix[user,co_item],
                                                                 uj=self.data_matrix[nei,co_item])
                    except IndexError:
                        self.sim_mat[user][nei] = 0
                        
                elif self.sim == 'os_sig':
                    try:
                        self.sim_mat[user][nei] = self.os_sig(ui=self.data_matrix[user,co_item],
                                                    uj=self.data_matrix[nei,co_item])
                    except IndexError:
                        self.sim_mat[user][nei] = 0
        
        # 유사도 행렬 lower triangle 부분 채워넣기.
        users_r = users[::-1]
        n=1
        for ui in users_r:
            neighbor = users_r[n:]
            n+=1
            for uj in neighbor:
                self.sim_mat[ui][uj] = self.sim_mat[uj][ui]



##############################################################################
    
    # 예측 평점 만들기.
    def predict(self):
        print()
        print('========================== predict =================================')
        print('========================== similarity:{}  k:{}========================='.format(self.sim, self.k))
        users = list(self.test_d.keys())
        error = []
        self.pred_check = []
        for user in tqdm(users):
            # k-neighbor
            k_neighbor_sim = sorted(self.sim_mat[user,:], reverse=True)[:self.k]
            k_neighbor_id = np.argsort(self.sim_mat[user,:])[::-1][:self.k]
            prediction=[]
            real=[]
            for no_rate_i, no_rate_r in self.test_d[user].items(): # 평점을 매기지 않은 아이템 중에서
                up=[]
                down=[]
                
                for nei_sim, nei_id in zip(k_neighbor_sim, k_neighbor_id): # k 이웃들에 대해
                    if no_rate_i in self.train_d[nei_id].keys(): # 평점을 매겼으면 예측에 사용.
                        up.append(nei_sim*(self.train_d[nei_id][no_rate_i]-self.data_mean[nei_id]))
                        down.append(nei_sim)
                try:
                    weight = sum(up)/sum(down)
                except:
                    weight = 0
                pred = self.data_mean[user] + weight
                if not math.isnan(pred):
                    prediction.append(pred)
                else:
                    prediction.append(self.data_mean[user])
                real.append(no_rate_r)
            
            self.pred_check.append(prediction)    
            e = [abs(i-j) for i,j in zip(prediction, real)]
            error.extend(e)
        
        mae = sum(error)/len(error)
        return mae

    
    def run(self):
        
        cv_result = []
        
        for i in range(self.CV):
            
            self.train_test_split()
            self.make_matrix()
            self.sigmoid_mapping()
            self.similarity_calculation()
            cv_result.append(self.predict())
        
        return np.mean(cv_result)
    

#%% experiment

CV=5

result_pcc_msd = {}
for sim in ['pcc','msd']:
    result_pcc_msd[sim]={}
    for k in list(range(10,101,10)):
        cf = CFwithKnn(data=data, data_d=data_d, k=k, CV=CV, sim=sim)
        result_pcc_msd[sim][k]=cf.run()


# save result
with open('result/result_{}_pcc_msd.pickle'.format(str(datetime.datetime.now())[:13]+'시'+str(datetime.datetime.now())[14:16]+'분'), 'wb') as f:
    pickle.dump(result_pcc_msd, f)

# load result
with open('result/result_2020-12-22 14시53분_traditional.pickle', 'rb') as f:
    result = pickle.load(f)




