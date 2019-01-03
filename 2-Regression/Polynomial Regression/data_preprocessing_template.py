# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]


# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

#creating regeression model


#prediction using polynomial linear regression
y_pred =  predict(regressor, data.frame(Level = 6.5))   
  
  
#creating visualizing for regression (for higher resolution and smooth curves)
library(ggplot2)
x_grid = seq(min(dataset$Level),max(dataset$Level),0.1)
 ggplot() +
   geom_point(aes(x = dataset$Level, y = dataset$Salary),
              colour = 'red') +
   geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
             colour = 'blue') +
   ggtitle('Truth or Bluff (Regression model)') +
   xlab('Level') +
   ylab('Salary')

 





# # Importing the libraries
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# 
# # Importing the dataset
# dataset = pd.read_csv('Position_Salaries.csv')
# X = dataset.iloc[:, 1:2].values
# y = dataset.iloc[:, 2].values
# 
# # Splitting the dataset into the Training set and Test set
# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# 
# # Feature Scaling
# """from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)"""
# 
# 
# #using the regression method 
# regressor
# 
# 
# #visualizing the information 
# X_grid = np.arange(min(X), max(X), 0.1)
# X_grid = X_grid.reshape((len(X_grid),1))
# plt.scatter(X,y,color = 'red')
# plt.plot(X_grid, regressor.predict(X), color ='blue' )
# plt.title('Truth or Bluff (polynomial regression)')
# plt.xlabel('position level')
# plt.ylabel('Salary')
# plt.show()
# 
# 
# 
# #using the regression to show prediction
# y_pred = regressor.predict(6.5)




# =============================================================================
# # Importing the libraries
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# 
# # Importing the dataset
# dataset = pd.read_csv('Data.csv')
# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, 3].values
# 
# # Splitting the dataset into the Training set and Test set
# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# 
# # Feature Scaling
# """from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)"""
# =============================================================================
