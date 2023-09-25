#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the dependencies
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.cluster import KMeans 


# In[2]:


import warnings

# Suppress specific warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 
warnings.simplefilter(action='ignore', category=UserWarning) 


# In[3]:


# DATA COLLECTION AND ANALYSIS
# Loading data from a CSV file into a pandas dataframe
customer_data = pd.read_csv(r"archive.csv")


# In[4]:


# First 5 rows in the dataframe
customer_data.head()


# In[5]:


# Finding number of rows and columns
customer_data.shape


# In[6]:


# Getting information about the dataset
customer_data.info() 


# In[7]:


# Checking for missing values
customer_data.isnull().sum() 


# In[8]:


# We choose the Annual income and spending score column for clustering
X = customer_data.iloc[:,[3,4]].values
print(X) 


# In[9]:


# Choosing the number of clusters
# WCSS = within clusters sum of squares
# finding WCSS value for different number of clusters

WCSS = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)


# In[10]:


# Visualizing the Elbow Method to find the optimal number of clusters
plt.figure(figsize=(10,5))
plt.plot(range(1, 11), WCSS, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# In[11]:


# Based on the Elbow Method, choose an optimal value of k (number of clusters)

# For example, if the plot suggests an "elbow" at k=5, you can choose k=5

# Running K-Means with the chosen value of k
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X) 


# In[12]:


# Visualizing the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4') 
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5') 
 
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers') 
plt.xlabel('Annual Income (k$)') 
plt.ylabel('Spending Score (1-100)')
plt.legend()


# In[ ]:




