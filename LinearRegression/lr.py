# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 12:25:24 2017

@author: SHRADDHA
"""

#from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

#Assign a styling to the graph
style.use('fivethirtyeight')


#Get two numpy arrays, they are the datasets to train our regression
#Instead of default we give values from csv
xs = np.array([1,3,4,5,6,7],dtype=np.float64)
ys = np.array([3,5,6,2,1,3],dtype=np.float64)


#try plotting the graph
'''plt.scatter(xs,ys)
plt.show()'''


#Since statistics library is not available in python2.7, made a function that returns mean
def mean(xs):
    sum = np.sum(xs)
    avg = sum / (len(xs))
    return avg

'''avg = mean(xs)
print(avg)'''
#y = mx+b


#Now we define a new function which takes our inputs/training sets, and gets a slope and b
#For all training set examples in csv we find the values
def best_fit_slope(xs,ys):
    m = ( (mean(xs) * mean(ys)) - (mean(xs*ys)) ) / ( ((mean(xs))**2) - (mean(xs**2)) )
    b = (mean(ys) - (m*mean(xs)) )
    return m,b
    

m,b = best_fit_slope(xs,ys)
print(m,b)


#Now that we have m and b, we can predict any value, this m and b is obtained by our training examples
#Used to predict
predict_x = 8 
predict_y = m*predict_x + b

plt.scatter(predict_x, predict_y, color='g')

regression_line = [(m*x)+b for x in xs]

                   #Plotting the Graph
plt.scatter(xs,ys)
plt.plot(xs,regression_line)
plt.show()               



