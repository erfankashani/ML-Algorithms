# Polynomial Regression

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]


# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

#predict the lm function for the best fit
lin_reg = lm(formula = Salary ~ Level ,
               data = dataset)
# you can type summary(regressor) on console to check the P value it should be less than 5% to show enough relationship between traing set data (we found a valid line )



#creating higher levels for the levels so we can do polynimal linear regeression
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4

poly_reg = lm(formula = Salary ~.,
              data = dataset,
              )

#creating visualizing for linear regression
library(ggplot2)
ggplot() +
   geom_point(aes(x = dataset$Level, y = dataset$Salary),
              colour = 'red') +
   geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
             colour = 'blue') +
   ggtitle('Truth or Bluff (Polynomial Regression)') +
   xlab('Level') +
   ylab('Salary')

#creating visualizing for polynomial linear regression
library(ggplot2)
 ggplot() +
   geom_point(aes(x = dataset$Level, y = dataset$Salary),
              colour = 'red') +
   geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
             colour = 'blue') +
   ggtitle('Truth or Bluff (Polynomial Regression)') +
   xlab('Level') +
   ylab('Salary')

 #  Visualising the Regression Model results (for higher resolution and smoother curve)
  library(ggplot2)
  x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
  ggplot() +
    geom_point(aes(x = dataset$Level, y = dataset$Salary),
               colour = 'red') +
    geom_line(aes(x = x_grid, y = predict(poly_reg,
                                          newdata = data.frame(Level = x_grid,
                                                               Level2 = x_grid^2,
                                                               Level3 = x_grid^3,
                                                               Level4 = x_grid^4))),
              colour = 'blue') +
    ggtitle('Truth or Bluff (Polynomial Regression)') +
    xlab('Level') +
    ylab('Salary')

#prediction using linear regression
 y_pred =  predict(lin_reg, data.frame(Level = 6.5))

#prediction using polynomial linear regression
 y_pred_poly =  predict(poly_reg, data.frame(Level = 6.5,
                                             Level2 = 6.5^2,
                                             Level3 = 6.5^3,
                                             Level4 = 6.5^4))
            Level4 = 6.5^4))
