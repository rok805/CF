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



def new_rating_sigmoid(data):

    for i in data:

        m = np.mean(list(data[i].values()))  # mean
        s = np.std(list(data[i].values()))  # std
        items = set(data[i].keys())

        #  ui mapping dictionary
        d = {}
        for x in items:
            y = 1 / (1 + np.exp(-data[i][x] + m)) - 0.5
            d[data[i][x]] = data[i][x] + y / (1 + s**2)

        # ui rating mapping
        for item in data[i]:
            data[i][item] = d[data[i][item]]

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

    similarity = 1 / abs(1+sum(dist))

    # pncr
    pncr = np.exp(-(N-c_length)/N)

    return similarity * pncr, c_length
