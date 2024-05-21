#importing the library
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

#importing the dataset 

dataset = pd.read_csv('c:\\Users\\DELL\\Desktop\\datasets\\Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#training the linear regression mocdel on the whole dataset 
from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(x,y)

#training the polynomial regression model on the whole dataset 

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2= LinearRegression()
lin_reg_2.fit(x_poly,y)

#visualising the linear regression result 
plt.scatter(x,y,color = 'red')
plt .plot (x,lin_reg.predict(x),color = 'blue')
plt.title ("Satya hai bhaiya (Linear Regression )")
plt.xlabel("positions")
plt.ylabel("salary")
plt.show()

#visualise the polynomial regression result 
plt.scatter(x,y,color = 'red')
plt .plot (x,lin_reg_2.predict(poly_reg.fit_transform(x)),color = 'blue')
plt.title ("Satya hai bhaiya (polynomial Regression )")
plt.xlabel("positions")
plt.ylabel("salary")
plt.show()

#predicting a new result with linear regression 

lin_reg.predict([[6.5]])
#predicting a new result with polynomial  regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))