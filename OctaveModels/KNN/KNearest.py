# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 22:37:59 2017

@author: SHRADDHA
"""

import pandas as pd
import numpy as np
from sklearn import cross_validation, neighbors

df = pd.read_csv('breastcancer.txt')
df.replace('?',-99999,inplace=True)

df.drop(['id'],1,inplace=True)

X =np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)


example_measures = np.array([4,2,1,1,1,2,3,2,1])
example_measures = example_measures.reshape(1,-1)

prediction = clf.predict(example_measures)
