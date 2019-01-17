# K-Means Clustering

# Importing the dataset
 dataset = read.csv('Mall_Customers.csv')
 X = dataset[4:5]

 #using the elbow method to find the number of clusters
 set.seed(6)
 wcss <- vector()
 for(i in 1:10) wcss[i] <- sum(kmeans(X,i)$withinss)
 plot(1:10, wcss , type = 'b' , main = paste('cluster of clients'), xlab = "Number of Clusters" , ylab = "wcss")

 #applying the model to the mall set
 set.seed(29)
 #Fitting K-Means to the dataset
kmeans = kmeans(x = dataset, centers = 5)
y_kmeans = kmeans$cluster

 #Visializing the cluster
 library(cluster)
 clusplot(X,
          kmeans$cluster,
          lines =0,
          shade = TRUE,
          color = TRUE,
          labels = 2,
          plotchar = FALSE,
          span = TRUE,
          main = paste('Cluster of clients'),
          xlab = "Anual income",
          ylab = "Spending Score")
