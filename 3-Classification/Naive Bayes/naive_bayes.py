# Naive Bayes

#import library
import numpy as np #for any mathematic codes
import matplotlib.pyplot as plt # help for ploting nice charts
import pandas as pd #import datasets and manage datasets

#import a dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, 2:4].values #takinf all the culums -1 and taking all the rows
y = dataset.iloc[:, 4].values

#spliting / making trainingset and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = 0.25, random_state = 0)

#feature scaling // for the Xtrain we have to do fit then transform but for xtest we only need the transform since its already fir cause of xtrain
 from sklearn.preprocessing import StandardScaler
 sc_X = StandardScaler()
 X_train = sc_X.fit_transform(X_train)
 X_test = sc_X.transform(X_test)


#fitting classification model to training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)


#predict the test set results
y_pred = classifier.predict(X_test)

#making the Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#visualizing the training set results
 # Visualising the Test set results
 from matplotlib.colors import ListedColormap
 X_set, y_set = X_train, y_train
 X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
 plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
 plt.xlim(X1.min(), X1.max())
 plt.ylim(X2.min(), X2.max())
 for i, j in enumerate(np.unique(y_set)):
     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                 c = ListedColormap(('red', 'green'))(i), label = j)
 plt.title('Baive Bay Classification (Train set)')
 plt.xlabel('Age')
 plt.ylabel('Estimated Salary')
 plt.legend()
 plt.show()

#visualizing the test set results
 # Visualising the Test set results
 from matplotlib.colors import ListedColormap
 X_set, y_set = X_test, y_test
 X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
 plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
 plt.xlim(X1.min(), X1.max())
 plt.ylim(X2.min(), X2.max())
 for i, j in enumerate(np.unique(y_set)):
     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                 c = ListedColormap(('red', 'green'))(i), label = j)
 plt.title('Baive Bayse Classification (Test set)')
 plt.xlabel('Age')
 plt.ylabel('Estimated Salary')
 plt.legend()
 plt.show()
