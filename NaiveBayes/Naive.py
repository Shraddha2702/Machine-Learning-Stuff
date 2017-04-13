# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 13:53:45 2017

@author: SHRADDHA
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.naive_bayes import GaussianNB 

style.use('fivethirtyeight')

xs = [[1,3,5,7,3,5]]
ys = [[2,5,7,2,1,5]]

#x_test = np.array([10,11,12,11,12],dtype=np.float64)

clf =  GaussianNB()
clf.fit(xs,ys)

plt.scatter(xs,ys)
#plt.plot(x_test,y_test)
plt.show()


