#!/usr/bin/env python
# coding: utf-8

# # Shaunak Pande

# ## Data Science & Business Analytics Task 1

# ## Prediction using Unsupervised Machine Learning

# In[ ]:





# In[3]:


# Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


# In[4]:


df=pd.read_csv("iris.csv")
df


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


#predict the optimun number of clusters


# In[6]:


x = df.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
y = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    y.append(kmeans.inertia_)


# In[7]:


#plotting the result


# In[9]:


plt.plot(range(1, 11), y)
#plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()


# In[ ]:





# In[ ]:





# In[10]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# In[ ]:





# In[11]:


# Visualising the clusters 
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




