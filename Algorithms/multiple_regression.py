#data preprocessing

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv("50_Startups.csv")   #f5 to set directory 
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X =  LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding the dummies varibale trap
X = X[:,1:]

#spliting the dataaset into training set and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#feature scaling
''' from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)  
X_test = sc_X.transform(X_test)    '''

#fitting the mutiple liear regression in the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)

#building the optimal model using backward elimination
import statsmodels.formula.api as sm                       
X = np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)    #for adding b0 as stats.mode.api ignore b0 therefore we are adding 1 to beginning to incude b0
X_opt = X[:,[0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()
X_opt = X[:,[0,1,3,4,5]]
regressor_ols = sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()
X_opt = X[:,[0,3,4,5]]
regressor_ols = sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()
X_opt = X[:,[0,3,5]]
regressor_ols = sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()
X_opt = X[:,[0,3]]
regressor_ols = sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()