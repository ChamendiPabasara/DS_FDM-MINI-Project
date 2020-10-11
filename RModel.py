import pandas as pd
import numpy as np
import pickle
from sklearn import datasets
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from math import sqrt


df = pd.read_csv("Cleaned_Fundraising.csv")
#df.head()

df.AVGGIFT=df['AVGGIFT'].round(decimals=2)
df.loc[df['zipconvert_3'] == 1, 'zipconvert_3'] = 2
df.loc[df['zipconvert_4'] == 1, 'zipconvert_4'] = 3
df.loc[df['zipconvert_5'] == 1, 'zipconvert_5'] = 4
df['zip'] = df['zipconvert_2'] + df['zipconvert_3'] + df['zipconvert_4'] + df['zipconvert_5']

df.AVGGIFT=df['AVGGIFT'].round(decimals=2)


X = df.drop(['TARGET_B','zipconvert_2','zipconvert_3','zipconvert_4','zipconvert_5','Row Id','Row Id.','TARGET_D'],axis=1)
y= df['TARGET_D']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=5)

scaler = StandardScaler()
train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.transform(X_test)

#Linear Regression
regressor = LinearRegression()
regressor.fit(train_scaled, y_train)


# Saving model to disk
pickle.dump(regressor, open('Linear_Regression_Model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('Linear_Regression_Model.pkl','rb'))
#print(model.predict([[2, 9, 6]]))

#mse = mean_squared_error(y_train, model.predict(train_scaled))
#mae = mean_absolute_error(y_train, model.predict(train_scaled))

#print("mse = ",mse," & mae = ",mae," & rmse = ", sqrt(mse))

#test_mse = mean_squared_error(y_test, model.predict(test_scaled))
#test_mae = mean_absolute_error(y_test, model.predict(test_scaled))
#print("mse = ",test_mse," & mae = ",test_mae," & rmse = ", sqrt(test_mse))

