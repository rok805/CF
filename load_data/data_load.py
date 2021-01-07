# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 22:16:52 2020

@author: user
"""



import pandas as pd


def create_rating():

    rating = pd.read_csv('movielens/order/u.data', sep='\t', header=None)
    rating.columns = ['userId','movieId','rating','timestamp']
    idx1 = [i-1 for i in rating['userId']] # userId 를 0부터 시작하게 함.
    idx2 = [i-1 for i in rating['movieId']] # movieId 를 0부터 시작하게 함.
    rating['userId'] = idx1
    rating['movieId'] = idx2

    return rating

def create_rating_1M():
    rating = pd.read_csv('movielens/1M/ratings.dat', sep='::', header=None)
    rating.columns = ['userId','movieId','rating','timestamp']
    idx1 = [i-1 for i in rating['userId']] # userId 를 0부터 시작하게 함.
    idx2 = [i-1 for i in rating['movieId']] # movieId 를 0부터 시작하게 함.
    rating['userId'] = idx1
    rating['movieId'] = idx2
    return rating


# change user rating matrix to dictionary
def create_rating_dic():

    rating = create_rating()
    rd = {}
    users = set(rating['userId'])
    for i in users: 
        tmp = rating[rating['userId']==i]
        rd[i] = {a:b for a,b in zip(tmp['movieId'],tmp['rating'])}

    return rd

def create_rating_dic_1M():

    rating = create_rating_1M()
    rd = {}
    users = set(rating['userId'])
    for i in users: 
        rd[i] = {}
        tmp = rating[rating['userId']==i]
        
        rd[i] = {a:b for a,b in zip(tmp['movieId'],tmp['rating'])}
        
    return rd

##############################################################################

def create_rating_netflix():

    rating = pd.read_csv(r'C:\Users\user\Documents\CF\netflix\movie_ratings\ratings.csv')
    rating.columns = ['userId','movieId','rating','timestamp']
    idx1 = [i-1 for i in rating['userId']] # userId 를 0부터 시작하게 함.
    idx2 = [i-1 for i in rating['movieId']] # movieId 를 0부터 시작하게 함.
    rating['userId'] = idx1
    rating['movieId'] = idx2

    return rating



# change user rating matrix to dictionary
def create_rating_dic_netflix():

    rating = create_rating_netflix()
    rd = {}
    users = set(rating['userId'])
    for i in users: 
        tmp = rating[rating['userId']==i]
        rd[i] = {a:b for a,b in zip(tmp['movieId'],tmp['rating'])}

    return rd


##############################################################################

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

