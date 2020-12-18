# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 13:16:24 2020

@author: user
"""

import numpy as np
import math


def cosine_similarity(ui: dict, uj: dict):  # ui, uj: dictionary

    ui_item = set(ui.keys())
    uj_item = set(uj.keys())
    corated_item = ui_item.intersection(uj_item)
    c_length = len(corated_item)

    if c_length == 0:  # replace 0 when there is no co-rated rating.
        return 0, c_length

    ui_cr = [ui[c] for c in corated_item]
    uj_cr = [uj[c] for c in corated_item]

    up = np.dot(ui_cr, uj_cr)
    down = (math.sqrt(sum([i ** 2 for i in ui_cr])) *
            math.sqrt(sum([j ** 2 for j in uj_cr])))

    return up/down, c_length


def PCC_similarity(ui: dict, uj: dict):

    ui_item = set(ui.keys())
    uj_item = set(uj.keys())
    corated_item = ui_item.intersection(uj_item)
    c_length = len(corated_item)

    if c_length == 0:  # replace 0 when there is no co-rated rating.
        return 0, c_length

    ui_mean = np.mean(list(ui.values()))
    uj_mean = np.mean(list(uj.values()))

    up = sum([(ui[c]-ui_mean) * (uj[c]-uj_mean) for c in corated_item])
    down = np.sqrt(sum([(ui[i]-ui_mean) ** 2 for i in corated_item])) \
        * np.sqrt(sum([(uj[j]-uj_mean) ** 2 for j in corated_item]))

    try:
        return up/down, c_length
    except ZeroDivisionError:
        return 0, c_length


def MSD_similarity(ui: dict, uj: dict):

    ui_item = set(ui.keys())
    uj_item = set(uj.keys())
    corated_item = ui_item.intersection(uj_item)
    c_length = len(corated_item)

    if c_length == 0:
        return 0, c_length

    ui_cr = [ui[c] for c in corated_item]
    uj_cr = [uj[c] for c in corated_item]

    up = sum([(i - j) ** 2 for i, j in zip(ui_cr, uj_cr)])
    down = c_length

    return 1 - up/down, c_length


def Jaccard_similarity(ui: dict, uj: dict):

    ui_item = set(ui.keys())
    uj_item = set(uj.keys())
    corated_item = ui_item.intersection(uj_item)
    c_length = len(corated_item)

    up = len(corated_item)
    down = len(ui_item.union(uj_item))

    return up/down, c_length


def os(ui: dict, uj: dict, N: int):

    # PNCR
    corated = set(ui.keys()).intersection(set(uj.keys()))

    c_length = len(corated)

    if c_length == 0:
        return 0, c_length

    pncr = np.exp(-(N-c_length)/N)

    # ADF

    basket = []

    for c in corated:
        basket.append(np.exp(- abs(ui[c] - uj[c]) / max(ui[c], uj[c])))
    adf = sum(basket) / c_length

    return pncr * adf, c_length


def os_new_rating(ui: dict, uj: dict, N: int):

    # PNCR
    corated = set(ui.keys()).intersection(set(uj.keys()))

    c_length = len(corated)

    if c_length == 0:
        return 0, c_length

    pncr = np.exp(-(N-c_length)/N)

    # ADF

    basket = []

    for c in corated:
        basket.append(np.exp(- abs(ui[c] - uj[c]) / max(abs(ui[c]), abs(uj[c]))))
    adf = sum(basket) / c_length

    return pncr * adf, c_length


def os_new_rating_2times(ui: dict, uj: dict, N: int):

    # PNCR
    corated = set(ui.keys()).intersection(set(uj.keys()))

    c_length = len(corated)

    if c_length == 0:
        return 0, c_length

    pncr = np.exp(-(N-c_length)/N)

    # ADF

    basket = []

    for c in corated:
        basket.append(np.exp(- abs(ui[c] - uj[c]) / (max(abs(ui[c]), abs(uj[c]))*2)))
    adf = sum(basket) / c_length

    return pncr * adf, c_length