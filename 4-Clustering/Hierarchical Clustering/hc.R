# Hierarchical Clustering

# Importing the dataset
dataset = read.csv('Mall_Customers.csv')
X = dataset[4:5]

#using the dendrogram to find the number of the clusters
dendrogram = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
plot(dendrogram,
     main = paste('Dendrogram'),
     xlab = 'Customer',
     ylab = 'Euclidean Distance')

#fit our hierchial cluster with 5 clusters since we found it above
hc = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
y_hc = cutree(hc, 5 )


#Visializing the cluster
library(cluster)
clusplot(X,
         y_hc,
         lines =0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Cluster of clients'),
         xlab = "Anual income",
         ylab = "Spending Score")
