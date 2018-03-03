import numpy as np
import urllib
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error,r2_score

url="http://goo.gl/j0Rvxq"
raw_data=urllib.urlopen(url)
df=np.loadtxt(raw_data,delimiter=",")
#Dat = np.genfromtxt(df, names=True)
#print(Dat)
print(df.shape)
#df.fillna(-9999,inplace=True)
X= df[:,0:8]
#X = df.data[:, np.newaxis, 2]
y=df[:,8]
X=np.delete(X,2,1)
X=preprocessing.scale(X)
#print X.feature_names
print(X.shape)
#print(X)
print(y.shape)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

model=LinearRegression(n_jobs=-1)
model.fit(X_train,y_train)

accuracy=model.score(X_test,y_test)
print(accuracy)

y_pred=model.predict(X_test)

print('Coefficients:\n',model.coef_)

print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))

print('Variance score: %.2f' % r2_score(y_test, y_pred))
print(X_test.shape)
print(y_test.shape)

plt.scatter(X_test,y_test,c='red')
plt.xlabel("features")
plt.ylabel("malign/beningn")
plt.title("diabetes linear regression")
plt.plot(X_test,y_pred,c='black',linewidths=2)

plt.show()
























