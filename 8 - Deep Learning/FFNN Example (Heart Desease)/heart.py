# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 07:51:06 2019
This Artificial Neuro Network Deep Learning model attempts to predict the heart desease 
in group of patients. The dataset has 12 dependent varibles 
The model was designed with 2 hidden layers with rectifier activation function, and the output layer using sigmoid activation function.  
the y-test validation shows 86.9% accuracy for the model 
@author: erfan Kashani
"""


#import library
import numpy as np #for any mathematic codes
import matplotlib.pyplot as plt # help for ploting nice charts 
import pandas as pd #import datasets and manage datasets

#import a dataset
dataset = pd.read_csv('heart.csv')
X = dataset.iloc[:, 0:13].values #takinf all the culums -1 and taking all the rows
y = dataset.iloc[:, 13].values 


#spliting / making trainingset and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = 0.20, random_state = 0)

#feature scaling // for the Xtrain we have to do fit then transform but for xtest we only need the transform since its already fir cause of xtrain
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Starting the ANN algorithem 
import keras

from keras.models import Sequential
from keras.layers import Dense

#initial the ANN
classifier  = Sequential()

#adding the input layer and the first hidden layer 
classifier.add(Dense(output_dim = 7,init = 'uniform' , activation = 'relu' , input_dim = 13 ))

#adding headen layers (using rectifier activation function // output dim as average of input and output parameters (11+1)/2 
classifier.add(Dense(output_dim = 7,init = 'uniform' , activation = 'relu' ))

#adding the output layer(using sigmoid activation function // if the output needed more than one category then use softmax activation function) 
classifier.add(Dense(output_dim = 1,init = 'uniform' , activation = 'sigmoid' ))

#combile the ANN
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'] )

#fitting ANN model to training set
classifier.fit(X_train, y_train, batch_size = 10 , nb_epoch = 100)


#create the classifier algorithem 

#predict the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
 
#making the Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)