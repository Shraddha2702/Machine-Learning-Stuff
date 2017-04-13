import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
#import datetime
import matplotlib.pyplot as plt
from matplotlib import style
#import time
import pickle 

#Styling of the Graph
style.use('ggplot')


#Use of Quandl DataSet for Prediction
quandl.ApiConfig.api_key = 'kEad-D-qUCWfpDwE7Jwx'

df = quandl.get_table('WIKI/PRICES')
df = df[['adj_open','adj_high','adj_low','adj_close','adj_volume']]


#New Features Created
df['PCT_HL'] = ((df['adj_high'] - df['adj_low'])/df['adj_close']) * 100.0
df['PCT_change'] = ((df['adj_close']-df['adj_open'])/df['adj_open']) * 100.0


#New List Created consisting of new Features
df = df[['adj_close','PCT_HL','PCT_change','adj_volume']]
#print(df.head)



#Time to forecast some values
forecast_col = 'adj_close'
df.fillna(-99999,inplace=True)

forecast_out = int(math.ceil(0.1*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
#df.dropna(inplace=True)
#print(df.head)
#print(df.tail)


X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
#print(X_lately)
X = X[:-forecast_out]

Y = np.array(df['label'])
Y = Y[:-forecast_out]
#df.dropna(inplace=True)

print(len(X))
print(len(Y))



#Train and Test the Dataset division
x_train, x_test, y_train, y_test = cross_validation.train_test_split(X,Y, test_size = 0.2)
 

#Demonstration of pickle, really useful since we don't need to train the data again and again, as pickle saves the state and the trained data can be used to see the results
#Classifier commented out since pickle is created
clf = LinearRegression()
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)

forecast_set = clf.predict(X_lately)

with open('linearreg.pickle','wb') as f:
    pickle.dump(clf,f)
    
pickle_in = open('linearreg.pickle','rb')
clf= pickle.load(pickle_in)
print(forecast_set, accuracy, forecast_out)


#Some Faults here 
df['Forecast'] = np.nan
'''last_date = df.iloc[-1].name
last_unix = time.mktime(datetime.datetime.strptime(str(last_date), "%Y-%m-%d %H:%M:%S").timetuple())
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]'''

print(df.head())
print(df.tail())


#Time to plot the Graph finally
df['adj_close'].plot()
df['Forecast'].plot()

plt.legend()
plt.xlabel('date')
plt.ylabel('price')
plt.show()


#print(accuracy)
#print(forecast_out)