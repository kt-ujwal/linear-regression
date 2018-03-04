from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')
xs = np.array([1,2,3,4,5,6,7],dtype=np.float64)
ys= np.array([5,4,6,5,6,7,6],dtype=np.float64)

#plt.scatter(xs,ys)
#plt.show()

def best_fit_slope_intercept(xs,ys):
	n1=((mean(xs)*mean(ys))-mean(xs*ys))
	d1=((mean(xs)*mean(xs))-mean(xs*xs))
	m=n1/d1
	b=mean(ys)-m*mean(xs)
	return m,b

def squared_error(ys_orig,ys_line):
	return sum((ys_line-ys_orig)**2)
	
def coefficient_of_determination(ys_orig,ys_line):
	y_mean_line=[mean(ys_orig) for y in ys_orig]
	squared_error_regr=squared_error(ys_orig,ys_line)
	squared_error_y_mean=squared_error(ys_orig,y_mean_line)
	return 1-(squared_error_regr/squared_error_y_mean)


m,b=best_fit_slope_intercept(xs,ys)
print(m,b)
regression_line=[]
#regression_line=[(m*x)+b for x in xs] predicting ys from m and b
for x in xs:
	regression_line.append((m*x)+b)
predict_x =np.array([2.5,6.7,3.4,8.1,0.4,1.2],dtype=np.float64)
predict_y=[(m*x)+b for x in predict_x]

r_squared=coefficient_of_determination(ys,regression_line)
print(r_squared)

plt.scatter(xs,ys,c='red')
plt.scatter(predict_x,predict_y,c='blue')
plt.plot(xs,regression_line,c='green',linewidth=2)
plt.show()





