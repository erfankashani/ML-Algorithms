# Data Preprocessing Template

#import library
import numpy as np #for any mathematic codes
import matplotlib.pyplot as plt # help for ploting nice charts
import pandas as pd #import datasets and manage datasets

#import a dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values #taking all the culums -1 and taking all the rows
y = dataset.iloc[:, 3].values

#managing the missing data
from sklearn.preprocessing import Imputer  #siket learnlibrary we getting imputer class
imputer = Imputer(missing_values = 'NaN',strategy = 'mean',axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#encode the categories // to put categories into numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#spliting / making trainingset and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = 0.2, random_state = 0)

#feature scaling // for the Xtrain we have to do fit then transform but for xtest we only need the transform since its already fir cause of
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
