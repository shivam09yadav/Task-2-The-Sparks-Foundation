#!/usr/bin/env python
# coding: utf-8

# ## Shivam Yadav
# ## Data Science and Business Analytics Intern(#GRIP July 2021) @ The Sparks Foundation.
# ## Prediction using Unsupervised ML.
# ## Predict the optimum number of clusters and represent it visually from the ‘Iris’ dataset.
# ## Dataset : https://bit.ly/3kXTdox.
# 

# # Importing the libraries:
# 

# In[1]:


import numpy as np
import pandas as pd
from sklearn import datasets
# Importing the visualization library:
import matplotlib.pyplot as plt

# Load the iris dataset
data = datasets.load_iris()
data_df = pd.DataFrame(data.data, columns = data.feature_names)
print("the first 5 rows")
data_df.head() 


# In[2]:


print("Checking the null values and the data type of each independent variable")
data_df.info()


# In[3]:


print("Getting a brief detail about the data")
data_df.describe()


# In[4]:


print("Dimension of the data")
data_df.shape


# In[5]:


print("x represents all the independent variables")
x = data_df.iloc[:, [0, 1, 2, 3]].values


# In[6]:


#Importing KMeans library:
from sklearn.cluster import KMeans


# # Creating the kmeans classifier randomly where k=5 :

# In[7]:


kmeans5 = KMeans(n_clusters = 5, init = 'k-means++',
                 max_iter = 300, n_init = 10, random_state = 0)
y_kmeans5 = kmeans5.fit_predict(x)
print(y_kmeans5)


# In[8]:


#Information about the cluster centres:
kmeans5.cluster_centers_


# # Finding the optimum number of clusters for k-means classification:

# In[9]:


wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
# Plotting the results onto a line graph, 
# `allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method for finding optimum number of clusters ')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()


# # From this we choose the number of clusters as '3'.
# 

# * Creating the kmeans classifier using elbow method where k = 3:

# In[10]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# In[11]:


#Information about the cluster centres:
kmeans.cluster_centers_


# # Visualising the clusters - On the first two columns:

# In[12]:


plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'yellow', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'grey', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'pink', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'red', label = 'Centroids')

plt.legend()

