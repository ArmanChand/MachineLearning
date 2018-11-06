#data preprocessing

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv("Position_Salaries.csv")   #f5 to set directory 
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,2].values

'''#spliting the dataaset into training set and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
'''

#feature scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)  
X_test = sc_X.transform(X_test)      '''

#fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#fitting polynomial reggression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X) 

lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)

#visualizing the linear regression
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('truth or bluff(Linear regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

#visualizing the polynomial linear regression
X_grid=np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X) ),color='blue')
plt.title('truth or bluff( polynomial Linear regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

#predicting with Linear reression
lin_reg.predict(6.5)

#predicting with polynomial regression
lin_reg_2.predict(poly_reg.fit_transform(6.5) )
