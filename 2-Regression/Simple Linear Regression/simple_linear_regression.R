# Simple Linear Regression

# Importing the dataset
dataset = read.csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary , SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)


#performing a simple Linear Regression model
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)
# you can type summary(regressor) on console to check the P value it should be less than 5% to show enough relationship between traing set data (we found a valid line )

# predicting the test value
y_pred = predict(regressor, newdata = test_set)

#visualize the results for the training set
install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience , y =training_set$Salary),
             colour = 'red') +
  geom_line(aes(x= training_set$YearsExperience , y= predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('salary vs. experience training set') +
  xlab('years of experience') +
  ylab('salary')

ggplot() +
  geom_point(aes(x = test_set$YearsExperience , y =test_set$Salary),
             colour = 'black') +
  geom_line(aes(x= training_set$YearsExperience , y= predict(regressor, newdata = training_set)),
            colour = 'red') +
  ggtitle('salary vs. experience testing set') +
  xlab('years of experience') +
  ylab('salary')
