# Apriori


#Dataprocessing
dataset = read.csv('Market_Basket_Optimisation.csv' , header = FALSE)
#install.packages('arules')
library(arules)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',' , rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN =10)


#training the Apiori on the dataset
rules = apriori(data = dataset, parameter = list(support = 0.004 , confidence = 0.2 ))

#visualizing the data
inspect(sort(rules, by = 'lift' )[1:10])
