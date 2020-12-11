# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:27:31 2020

@author: user
"""

import numpy as np
from collections import Counter

# euclidean distance
def euclidean_dist(a, b):
    result = 0
    for i in zip(a, b):
        result += (i[0] - i[1])**2

    return np.sqrt(result)

# change rating to sigmoid_mapping fuction vlaue
# mid means rating which is 'so-so'


def sigmoid_mapping_d(ratings_list, mid):

    sigmoid_dic = {}

    for i in ratings_list:
        sigmoid_dic[i] = (1 + mid**2) / (1 + np.exp(-i + mid))

    return sigmoid_dic


def sigmoid_mapping_d_1(ratings_list, mid):

    sigmoid_dic = {}

    for i in ratings_list:
        sigmoid_dic[i] = (1 + mid) / (1 + np.exp(-i + mid))

    return sigmoid_dic


def sigmoid_mapping_d_2(ratings_list, mid):

    sigmoid_dic = {}

    for i in ratings_list:
        sigmoid_dic[i] = (1 + np.sqrt(mid)) / (1 + np.exp(-i + mid))

    return sigmoid_dic


def sigmoid_mapping_d_3(ratings_list, mid):

    sigmoid_dic = {}

    for i in ratings_list:
        sigmoid_dic[i] = 1 / (1 + np.exp(-i + mid))

    return sigmoid_dic



def sigmoid_mapping_d2(ratings_list, mid):

    sigmoid_dic = {}

    for i in ratings_list:
        sigmoid_dic[i] = np.exp(-np.exp(-i+mid)) * (mid ** 2)

    return sigmoid_dic


def sigmoid_mapping_d2_1(ratings_list, mid):

    sigmoid_dic = {}

    for i in ratings_list:
        sigmoid_dic[i] = np.exp(-np.exp(-i+mid)) * mid

    return sigmoid_dic


def sigmoid_mapping_d2_2(ratings_list, mid):

    sigmoid_dic = {}

    for i in ratings_list:
        sigmoid_dic[i] = np.exp(-np.exp(-i+mid))

    return sigmoid_dic


def sigmoid_mapping_d3(ratings_list, mid):

    sigmoid_dic = {}

    for i in ratings_list:
        sigmoid_dic[i] = (1 - np.exp(-np.exp(i-mid))) * mid**2

    return sigmoid_dic


def sigmoid_mapping_d3_1(ratings_list, mid):

    sigmoid_dic = {}

    for i in ratings_list:
        sigmoid_dic[i] = (1 - np.exp(-np.exp(i-mid))) * mid

    return sigmoid_dic


def sigmoid_mapping_d3_2(ratings_list, mid):

    sigmoid_dic = {}

    for i in ratings_list:
        sigmoid_dic[i] = (1 - np.exp(-np.exp(i-mid)))

    return sigmoid_dic


def new_rating_mean_1(data):

    for i in data:  # users
        m = np.mean(list(data[i].values()))  # user's mean

        for item in data[i]:  # user i's items
            data[i][item] = round((data[i][item] - m) / m, 5)

    return data


def new_rating_mean_std_2(data):

    for i in data:  # users
        m = np.mean(list(data[i].values()))  # user's mean
        s = np.std(list(data[i].values()))  # user's std

        for item in data[i]:  # user i's items
            data[i][item] = round(1 / (1 + np.exp(- data[i][item] + m)) - 0.5, 5)

    return data


def new_rating_mean_std_3(data):

    for i in data:  # users
        m = np.mean(list(data[i].values()))  # user's mean
        s = np.std(list(data[i].values()))  # user's std

        for item in data[i]:  # user i's items
            data[i][item] = round(1 / (1 + np.exp(- data[i][item] + m)) - 0.5, 5) / s**2

    return data


def sigmoid_mapping_d_pf(ratings_list, mid):

    sigmoid_dic = {}

    for i in ratings_list:
        sigmoid_dic[i] = (1 - np.exp(-np.exp(i-mid))) * mid

    return sigmoid_dic


def os_sig(ui, uj, N, sigmoid_dic):

    # PNCR
    corated_item = set(ui.keys()).intersection(set(uj.keys()))
    c_length = len(corated_item)

    if c_length == 0:  # assign 0 when there is no co-rated rating.
        return 0, c_length

    pncr = np.exp(-(N-c_length)/N)

    # ADF
    dist = []
    for c in corated_item:
        a = (ui[c], sigmoid_dic[ui[c]])
        b = (uj[c], sigmoid_dic[uj[c]])

        dist.append(np.exp(-euclidean_dist(a, b)))

    adf = sum(dist) / c_length

    return pncr * adf, c_length


def os_sig_max(ui, uj, N, sigmoid_dic):

    # PNCR
    corated_item = set(ui.keys()).intersection(set(uj.keys()))
    c_length = len(corated_item)

    if c_length == 0:  # replace 0 when there is no co-rated rating.
        return 0, c_length

    pncr = np.exp(-(N-c_length)/N)

    # ADF
    dist = []
    for c in corated_item:
        a = (ui[c], sigmoid_dic[ui[c]])
        b = (uj[c], sigmoid_dic[uj[c]])

        dist.append(np.exp(-euclidean_dist(a, b)) / max(ui[c], uj[c]))

    adf = sum(dist) / c_length

    return pncr * adf, c_length

def os_sig_no_max(ui, uj, N, sigmoid_dic, max_r):

    # PNCR
    corated_item = set(ui.keys()).intersection(set(uj.keys()))
    c_length = len(corated_item)

    if c_length == 0:  # replace 0 when there is no co-rated rating.
        return 0, c_length

    pncr = np.exp(-(N-c_length)/N)

    # ADF
    dist = []
    for c in corated_item:
        a = (ui[c], sigmoid_dic[ui[c]])
        b = (uj[c], sigmoid_dic[uj[c]])

        dist.append(np.exp(- (euclidean_dist(a, b) / max_r) ))

    adf = sum(dist) / c_length

    return pncr * adf, c_length


def os_sig_no_euclidean(ui, uj, N, sigmoid_dic, max_r):

    # PNCR
    corated_item = set(ui.keys()).intersection(set(uj.keys()))
    c_length = len(corated_item)

    if c_length == 0:  # replace 0 when there is no co-rated rating.
        return 0, c_length

    pncr = np.exp(-(N-c_length)/N)

    # ADF
    dist = []
    for c in corated_item:

        dist.append(np.exp(- (abs(ui[c] - uj[c]) / max(ui[c], uj[c])) ))

    adf = sum(dist) / c_length

    return pncr * adf, c_length


def os_sig_no_euclidean_no_max(ui, uj, N, sigmoid_dic, max_r):

    # PNCR
    corated_item = set(ui.keys()).intersection(set(uj.keys()))
    c_length = len(corated_item)

    if c_length == 0:  # replace 0 when there is no co-rated rating.
        return 0, c_length

    pncr = np.exp(-(N-c_length)/N)

    # ADF
    dist = []
    for c in corated_item:

        dist.append(np.exp(- (abs(ui[c] - uj[c]) / max_r) ))

    adf = sum(dist) / c_length

    return pncr * adf, c_length


# distance between sigmoid mapping values
def sigmoid_mapping_similarity(ui, uj, N, sigmoid_dic):

    ui_item = set(ui.keys())
    uj_item = set(uj.keys())
    corated_item = ui_item.intersection(uj_item)
    c_length = len(corated_item)

    if c_length == 0:  # replace 0 when there is no co-rated rating.
        return 0, c_length

    dist = []
    for c in corated_item:
        a = (ui[c], sigmoid_dic[ui[c]])
        b = (uj[c], sigmoid_dic[uj[c]])

        dist.append(euclidean_dist(a, b))

    similarity = 1 / (1+sum(dist))**2

    # pncr
    pncr = np.exp(-(N-c_length)/N)

    return similarity * pncr, c_length


def pref_ratio(data):

    rd_pref = {}
    for i in data:
        rd_pref[i] = {
            'pos': 0,
            'neg': 0,
            'neu': 0,
            'ratio': 0,
            'exp': 0
            }
        pref = dict(Counter(data[i].values()))
        for j in pref:
            if j > 3.0:
                rd_pref[i]['pos'] += pref[j]
            elif j == 3.0:
                rd_pref[i]['neu'] += pref[j]
            else:
                rd_pref[i]['neg'] += pref[j]
        rd_pref[i]['ratio'] = rd_pref[i]['pos'] / sum(pref.values())
        rd_pref[i]['exp'] = 1/(1+np.exp(-rd_pref[i]['ratio']))

    return rd_pref

