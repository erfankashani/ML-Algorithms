# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 21:43:37 2018
template for data pre processing credit to "machine learning A-Z course" https://www.superdatascience.com/machine-learning/
-Imports libraries.
-Imports the dataset.
-Spliting it into training-set and testing set.
-bonus: can scale the numbers also for specific machine learnin processes.
"""
# Data Preprocessing Template

#import library
import numpy as np #for any mathematic codes
import matplotlib.pyplot as plt # help for ploting nice charts
import pandas as pd #import datasets and manage datasets

#import a dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values #takinf all the culums -1 and taking all the rows
y = dataset.iloc[:, 3].values


#spliting / making trainingset and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = 0.2, random_state = 0)


#feature scaling // for the Xtrain we have to do fit then transform but for xtest we only need the transform since its already fir cause of xtrain
# =============================================================================
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
#
# =============================================================================
