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
from sklearn.naive_bayes import GaussianNB


df = pd.read_csv("Cleaned_Fundraising.csv")
#df.head()

df.AVGGIFT=df['AVGGIFT'].round(decimals=2)
df.loc[df['zipconvert_3'] == 1, 'zipconvert_3'] = 2
df.loc[df['zipconvert_4'] == 1, 'zipconvert_4'] = 3
df.loc[df['zipconvert_5'] == 1, 'zipconvert_5'] = 4
df['zip'] = df['zipconvert_2'] + df['zipconvert_3'] + df['zipconvert_4'] + df['zipconvert_5']

df.AVGGIFT=df['AVGGIFT'].round(decimals=2)


X = df.drop(['TARGET_B','zipconvert_2','zipconvert_3','zipconvert_4','zipconvert_5','Row Id','Row Id.','TARGET_D'],axis=1)
y= df['TARGET_B']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=5)

scaler = StandardScaler()
train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.transform(X_test)

clf = GaussianNB()

clf.fit(train_scaled,y_train)
score=clf.score(test_scaled, y_test)

# Saving model to disk
pickle.dump(clf, open('NaiveBayes_Classifier_Model.pkl','wb'))

# Loading model to compare the results
cmodel = pickle.load(open('NaiveBayes_Classifier_Model.pkl','rb'))


