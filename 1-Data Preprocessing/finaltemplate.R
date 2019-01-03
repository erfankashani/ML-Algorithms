# Data Preprocessing
#' Created on Thu Dec 27 21:43:37 2018
#' template for data pre processing
#' -Imports libraries
#' -Imports the dataset
#' -Spliting it into training-set and testing set
#' -bonus: can scale the numbers also for specific machine learnin processes
#' @author: erfan

dataset = read.csv('Data.csv')
# dataset = dataset[, 2:3]

##make training and testing sets
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased,
                     SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


##feature scaling

# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])
