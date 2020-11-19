# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 23:12:35 2020

@author: user
"""

from load_data import load_data
from collections import Counter
import matplotlib.pyplot as plt

rating = load_data.create_rating()
rating2 = load_data.create_rating_1m()
rating2.head()
rating2.columns=['userId','movieId','rating','timestamp']


rating_count = Counter(rating2['rating'])
rating_count = sorted(rating_count.items(), key = lambda x: x[0], reverse = True)


x = [i[0] for i in rating_count]
y = [i[1] for i in rating_count]


plt.bar(x=x, height=y, color='grey', width=0.3)
for i,j in zip(x, y):
    plt.text(x=i-0.2, y=j+300, s='{}%'.format(round(j/sum(y)*100,2)))
plt.title('ratio of ratings')
plt.ylabel('count')
plt.xlabel('rating')

len(set(rating2['userId']))



movie_count = dict(Counter(rating['movieId'])) # 영화당 평가 받은 rating의 개수.
movie_count = sorted(movie_count.items(), key = lambda x: x[1], reverse=False)

mc = Counter([i[1] for i in movie_count])

# 영화당 rating을 매긴 user의 사용자 수가 너무 적은 경우가 많음.
mc_x = list(mc.keys())
mc_y = list(mc.values())
plt.plot(mc_x, mc_y, ls='' ,marker = '.', markersize=2)
plt.xlabel('the number of rating')
plt.ylabel('frequency')
plt.title('rating freq per movies')

mc_y[0]/sum(mc_y)

tmp = rating['rating'].groupby(rating['userId']).mean()
plt.hist(tmp,width=0.2, color='grey', bins=10)
plt.title("user's average rating  0.1M data")
plt.xlabel('average of rating')
plt.ylabel('frequency')

tmp = rating2['rating'].groupby(rating2['userId']).mean()
plt.hist(tmp,width=0.2, color='grey')
plt.title("user's average rating  1M data")
plt.xlabel('average of rating')
plt.ylabel('frequency')



[{{1,2,3},{2,1},{1,2,4,3},{2}}]
s = "{{1,2,3},{2,1},{1,2,4,3},{2}}"
s = s[1:len(s)-1]
s = eval('['+s+']')

(1,3)[0] = 4
