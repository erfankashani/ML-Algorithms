# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the Regression Model to the dataset
# Create your regressor here

# Predicting a new result
y_pred = regressor.predict(6.5)

# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



# # Importing the dataset
# dataset = read.csv('Position_Salaries.csv')
# dataset = dataset[2:3]
# 
# # Splitting the dataset into the Training set and Test set
# # # install.packages('caTools')
# # library(caTools)
# # set.seed(123)
# # split = sample.split(dataset$Salary, SplitRatio = 2/3)
# # training_set = subset(dataset, split == TRUE)
# # test_set = subset(dataset, split == FALSE)
# 
# # Feature Scaling
# # training_set = scale(training_set)
# # test_set = scale(test_set)
# 
# # Fitting Decision Tree Regression to the dataset
# # install.packages('rpart')
# library(rpart)
# regressor = rpart(formula = Salary ~ .,
#                   data = dataset,
#                   control = rpart.control(minsplit = 1))
# 
# # Predicting a new result with Decision Tree Regression
# y_pred = predict(regressor, data.frame(Level = 6.5))
# 
# # Visualising the Decision Tree Regression results (higher resolution)
# # install.packages('ggplot2')
# library(ggplot2)
# x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
# ggplot() +
#   geom_point(aes(x = dataset$Level, y = dataset$Salary),
#              colour = 'red') +
#   geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
#             colour = 'blue') +
#   ggtitle('Truth or Bluff (Decision Tree Regression)') +
#   xlab('Level') +
#   ylab('Salary')
# 
# # Plotting the tree
# plot(regressor)
# text(regressor)