#data preprocessing

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv("Salary_Data.csv")   #f5 to set directory 
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#spliting the dataaset into training set and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)

#feature scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)  
X_test = sc_X.transform(X_test)      '''

#FITTING SIMPLE LINEAR REGRESSION TO THE TRAINING SET
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)

#visualising the training set reults
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('salary vs experience (training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

#visualising the test set reults
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('salary vs experience (test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()