import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing,svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')#plotting linear fun on graph




quandl.ApiConfig.api_key = '9xFuddvLhgsWyQGM7Mn3'
df =quandl.get("WIKI/GOOGL")#getting data from quandl:google stock prices

#print(df.head())#checking the data we're working with

df=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['hl_pct']=(df['Adj. High']-df['Adj. Low'])/df['Adj. Low']*100.0#high to low %
df['pct_change']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0#percentage change in stock prices
df=df[['Adj. Close','hl_pct','pct_change','Adj. Volume']]#reconfiguring our data with new attributes

#print(df.head())

forecast_col = 'Adj. Close'#to create label
df.fillna(-9999,inplace=True)#filling missing by making them outliers

forecast_out= int(math.ceil(0.005*len(df)))#forecasting todays stock from previous 18 days stocks
#print(len(df))
#print(0.005*len(df))
#print(forecast_out)#18=no. of days into future
df['label']=df[forecast_col].shift(-forecast_out)#shifting forecast_out up to align with previous days forecast_col

#print(df.head()) 

X = np.array(df.drop(['label'],1))# inputfeatureset
X = preprocessing.scale(X)#scaling X 
X_lately = X[-forecast_out:]#predicting m &c for X's with no (y)labels y=mX+c
X=X[:-forecast_out:]
df.dropna(inplace=True)

y= np.array(df['label'])#label

#print(len(X),len(y))#checking whether each feature set has label for training
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

model =LinearRegression(n_jobs=-1)
#model =svm.SVR(kernel='poly')
model.fit(X_train,y_train)

accuracy=model.score(X_test,y_test)#accuracy here is mean squared error
#print(accuracy*100.0)

forecast_set =model.predict(X_lately)#predicting stocks for next 18 days

print(forecast_set,accuracy,forecast_out)

df['Forecast'] =np.nan

last_date= df.iloc[-1].name#getting the name of last day(given by -1)
last_unix=last_date.timestamp()#last day
one_day =86400#no. of seconds per day
next_unix=last_unix+one_day#next day

for i in forecast_set:#populating our df with new dates
	next_date =datetime.datetime.fromtimestamp(next_unix)
	next_unix+=one_day
	df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)] +[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()









