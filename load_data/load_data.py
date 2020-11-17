# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 02:07:30 2020

@author: user
"""


import pandas as pd


def create_rating():
        
    rating = pd.read_csv('movielens/order/ratings.csv', sep=',', header=0)
    return rating

def create_rating_1m():
    rating = pd.read_csv('movielens/1M/ratings.dat', sep='::', header=None)
    return rating


# change user rating matrix to dictionary
def user_item_dictionary():
    
    rating = create_rating()
    rd = {}
    users = set(rating['userId'])
    for i in users: 
        rd[i] = {}
        tmp = rating[rating['userId']==i]
        
        rd[i] = {a:b for a,b in zip(tmp['movieId'],tmp['rating'])}
        
    return rd



# item genre data load
def create_genre():
    
    genre = pd.read_csv('movielens/order/genre.txt', sep='|')    
    genre.columns = ['genre','g_index']
    return genre



# item information data load
def create_item():
    
    item = pd.read_csv('movielens/order/item.txt', sep='|')    
    item.iloc[0:2,:]
    c = ['movie id', 'movie title', 'release date', 'video release date',
              'IMDb URL', 'unknown', 'Action' ,'Adventure ' ,'Animation',
              '''Children's''', 'Comedy', 'Crime' ,'Documentary' ,'Drama','Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
              'Thriller', 'War', 'Western']
    item.columns = c
    item = pd.concat([item[['movie id','movie title']],item.loc[:,'unknown':'Western']], axis=1)
    return item

