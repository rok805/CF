# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 13:16:24 2020

@author: user
"""

import numpy as np
import math

def cosine_similarity(a,b): #a,b: list type # must be co-rated

    if len(a) == 0:
        return 0
    
    up=np.dot(a,b)
    down=(math.sqrt(sum([i**2 for i in a])) * math.sqrt(sum([j**2 for j in b])))

    return up/down

def PCC_similarity(a,b): #a,b: list type # must be co-rated

    if len(a) == 0:
        return 0
    
    a_mean=np.mean(a) # average of a
    b_mean=np.mean(b) # average of b

    up=np.dot([i-a_mean for i in a],[j-b_mean for j in b])
    down=(math.sqrt(sum([(i-a_mean)**2 for i in a])) * math.sqrt(sum([(j-b_mean)**2 for j in b])))

    if down == 0.0:
        return 0
    return up/down

def MSD_similarity(a,b): #a,b: list type # must be co-rated

    if len(a) == 0:
        return 0
    
    dif = sum([(a-b)**2 for a,b in zip(a,b)])
    
    return abs(1 - dif/len(a))

def Jaccard_similarity(a,b): #a,b # must be whole rating respectively

    if len(a) == 0:
        return 0
    
    a = set(a)
    b = set(b)
    
    return len(a.intersection(b)) / len(a.union(b))



