# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
#always make sure they are matrixes some training sets are sensitive to this  
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# Fitting the SVR Model to the dataset

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')  # most common one is 'rbf' instead of 'poly'
regressor.fit(X,y)

#prediction based on the SVR model
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]) ) )) # here due to the feature scaling we are first scalling down the 6.5 and taking the prediction then taking the inverse scalling of the result and making meaningful result out of it
#transform function needed a array or matrix which we used numpy library arrat to create it, 

# Visualising the SCR Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()





# =============================================================================
# 
#  # Importing the libraries
#  import numpy as np
#  import matplotlib.pyplot as plt
#  import pandas as pd
#  
#  # Importing the dataset
#  dataset = pd.read_csv('Position_Salaries.csv')
#  X = dataset.iloc[:, 1:2].values
#  y = dataset.iloc[:, 2:3].values
#  
#  # Splitting the dataset into the Training set and Test set
#  """from sklearn.cross_validation import train_test_split
#  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""
# 
#  # Feature Scaling
#  from sklearn.preprocessing import StandardScaler
#  sc_X = StandardScaler()
#  sc_y = StandardScaler()
#  X = sc_X.fit_transform(X)
#  y = sc_y.fit_transform(y)
#  
#  # Fitting SVR to the dataset
#  from sklearn.svm import SVR
#  regressor = SVR(kernel = 'rbf')
#  regressor.fit(X, y)
# 
#  # Predicting a new result
#  y_pred = regressor.predict(6.5)
#  y_pred = sc_y.inverse_transform(y_pred)
#  
#  # Visualising the SVR results
#  plt.scatter(X, y, color = 'red')
#  plt.plot(X, regressor.predict(X), color = 'blue')
# plt.title('Truth or Bluff (SVR)')
#  plt.xlabel('Position level')
#  plt.ylabel('Salary')
#  plt.show()
#  
#  # Visualising the SVR results (for higher resolution and smoother curve)
#  X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
#  X_grid = X_grid.reshape((len(X_grid), 1))
#  plt.scatter(X, y, color = 'red')
#  plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
#  plt.title('Truth or Bluff (SVR)')
#  plt.xlabel('Position level')
#  plt.ylabel('Salary')
#  plt.show()
# 
# =============================================================================
