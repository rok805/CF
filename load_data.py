# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 02:07:30 2020

@author: user
"""

#%%
import pandas as pd

#%%
def create_rating():
        
    rating = pd.read_csv('ratings.csv', sep=',', header=0)
    return rating

#%%    
def create_genre():
    
    genre = pd.read_csv('genre.txt', sep='|')    
    genre.columns = ['genre','g_index']
    return genre

#%%
def create_item():
    
    item = pd.read_csv('item.txt', sep='|')    
    item.iloc[0:2,:]
    c = ['movie id', 'movie title', 'release date', 'video release date',
              'IMDb URL', 'unknown', 'Action' ,'Adventure ' ,'Animation',
              '''Children's''', 'Comedy', 'Crime' ,'Documentary' ,'Drama','Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
              'Thriller', 'War', 'Western']
    item.columns = c
    item = pd.concat([item[['movie id','movie title']],item.loc[:,'unknown':'Western']], axis=1)
    return item
#%%
