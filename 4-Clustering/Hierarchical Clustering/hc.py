# Hierarchical Clustering

#import library
import numpy as np #for any mathematic codes
import matplotlib.pyplot as plt # help for ploting nice charts
import pandas as pd #import datasets and manage datasets

#import a dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values #takinf all the culums -1 and taking all the rows

#use dendrogram to find the the optimal number of clusters
import scipy.cluster.hierarchy as sch
# creating the dendrogram // with linkage method which is the hierchial clustering // using wand method which minimizes the varience within each cluster
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title("Dendrogram")
plt.xlabel("Customer")
plt.ylabel("Euclidean Distances")
plt.show()

#fit the finding number of cluster to the HC clustering
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5 , affinity ='euclidean' , linkage = 'ward')
y_hc = hc.fit_predict(X)

#visualizing the results of the clusters (only works for two dimensions)
plt.scatter(X[y_hc ==0, 0],X[y_hc ==0,1], s =100, c ='red' , label = 'careful')
plt.scatter(X[y_hc ==1, 0],X[y_hc ==1,1], s =100, c ='blue' , label = 'standard')
plt.scatter(X[y_hc ==2, 0],X[y_hc ==2,1], s =100, c ='green' , label = 'Target')
plt.scatter(X[y_hc ==3, 0],X[y_hc ==3,1], s =100, c ='cyan' , label = 'careless')
plt.scatter(X[y_hc ==4, 0],X[y_hc ==4,1], s =100, c ='magenta' , label = 'Sensible')
plt.title('Cluster of Customers')
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending score(1-100)')
plt.legend()
plt.show()
