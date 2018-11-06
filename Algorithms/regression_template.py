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

#fitting reggression model to the dataset

#create your regressor here


#predicting the regression results
y_pred = regressor.predict(6.5) 


#visualizing the regression model (higher accuracy)
X_grid=np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('truth or bluff(regression model)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()


