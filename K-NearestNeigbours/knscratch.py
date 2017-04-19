# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 23:20:55 2017

@author: SHRADDHA
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter

style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}

#Two Classes and features

new_features = [5,7]



def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('k is set to a value less than total voting groups !')

    distances = []

    for group in data:
        for features in data[group]:
            #eucledian_distance = sqrt((features[0]-predict[0])**2 + (features[1]-predict[1])**2)
            #Above is hardcoded
            #Made available for everytime of code
            eucledian_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([eucledian_distance,group])

        votes = [i[1] for i in sorted(distances)[:k]]
        vote_result = Counter(votes).most_common(1)[0][0]
        print(vote_result)
        return vote_result

        
results = k_nearest_neighbors(dataset, new_features, k =3)
print(results)


for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0],ii[1], s=100, color=i)

plt.scatter(new_features[0],new_features[1],s=results)
plt.show()


