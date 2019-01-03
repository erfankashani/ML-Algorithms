# Multiple Linear Regression

# Importing the dataset
dataset = read.csv('50_Startups.csv')

#endocing the categories
dataset$State = factor(dataset$State ,
                         levels = c('New York', 'California' , 'Florida'),
                         labels = c(1,2,3))


# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit , SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#predict the lm function for the best fit
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State ,
               data = training_set)
# you can type summary(regressor) on console to check the P value it should be less than 5% to show enough relationship between traing set data (we found a valid line )

#predict the value on the testing set
y_pred =  predict(regressor, newdata = test_set)


# thimg to do // take care of b0 or the intercept // take care of the dummy varible trap  by removing one of the states

#optimal version of linear regression using the backward elimination method
regressor_opt = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend ,
               data = training_set)
regressor_opt = lm(formula = Profit ~ R.D.Spend ,
                   data = training_set)
y_pred_opt = predict(regressor_opt, newdata = test_set)



# #automating backward elimination using this formula
# backwardElimination <- function(x, sl) {
#   numVars = length(x)
#   for (i in c(1:numVars)){
#     regressor = lm(formula = Profit ~ ., data = x)
#     maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
#     if (maxVar > sl){
#       j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
#       x = x[, -j]
#     }
#     numVars = numVars - 1
#   }
#   return(summary(regressor))
# }
#
# SL = 0.05
# dataset = dataset[, c(1,2,3,4,5)]
# backwardElimination(training_set, SL)
