# -*- coding: utf-8 -*-
"""
Created on Wed May 14 23:06:59 2017

@author: SHRADDHA
"""

#import Section
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Part 1 DATA PREPROCESSING
dataset = pd.read_csv('train1.csv')
X = dataset.iloc[:,[1,16,17,18,29,78]].values
y = dataset.iloc[:,80].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:,4] = labelencoder_X_2.fit_transform(X[:, 4])
labelencoder_X_3 = LabelEncoder()
X[:,5] = labelencoder_X_3.fit_transform(X[:, 5])

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X)

import keras
from keras.models import Sequential #To Select model and train
from keras.layers import Dense #For Difference layers

classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(units=6, kernel_initializer = "uniform", 
                     activation = "relu", input_dim=6))

#Adding the second hidden layer same for all, only input not required
classifier.add(Dense(units=6, kernel_initializer = "uniform", 
                     activation = "relu"))

classifier.add(Dense(units=6, kernel_initializer = "uniform", 
                     activation = "relu"))

#Adding the final layer or Output layer
classifier.add(Dense(units=1, kernel_initializer = "uniform", 
                     activation = "sigmoid"))


classifier.compile(loss='mean_squared_error', optimizer='sgd',
                  metrics = ['accuracy'])



classifier.fit(X_train,y, batch_size=10 ,epochs = 100)