# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:12:26 2017

@author: SHRADDHA
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.metrics import accuracy_score

style.use('fivethirtyeight')

xs = [1,2,3,4,5]
ys = [3,5,7,8,9]

x_test = [3,5,10,12,14]

clf = LinearRegression()
y_test=clf.fit(np.transpose(np.matrix(xs)), np.transpose(np.matrix(ys))).predict(np.transpose(np.matrix(x_test)))

print(y_test)
#accuracy_score(np.transpose(np.matrix(ys)),y_test)
plt.scatter(xs,ys)
plt.plot(x_test,y_test, color='g')
plt.show()