#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("D:\Downloads/players_20.csv")


# In[3]:


df.head()


# In[4]:


df.describe().columns


# In[5]:


df = df[['short_name', 'age', 'international_reputation']]


# In[6]:


df.head()


# In[7]:


df = df[df.international_reputation > 3] # extracting players with overall above 86
df


# In[8]:



df.isnull().sum()


# In[9]:


df = df.fillna(df.mean())


# In[10]:


df.isnull().sum()


# In[11]:


names = df.short_name.tolist()
df = df.drop(['short_name'],axis = 1)


# In[12]:


df.head()


# In[13]:


from sklearn import preprocessing
x = df.values # numpy array
scaler = preprocessing.MinMaxScaler()
x_scaled = scaler.fit_transform(x)
X_norm = pd.DataFrame(x_scaled)


# In[14]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 2) # 2D PCA for the plot
reduced = pd.DataFrame(pca.fit_transform(X_norm))


# In[15]:


from sklearn.cluster import KMeans
# specify the number of clusters
kmeans = KMeans(n_clusters=6)
# fit the input data
kmeans = kmeans.fit(reduced)
# get the cluster labels
labels = kmeans.predict(reduced)
# centroid values
centroid = kmeans.cluster_centers_
# cluster values
clusters = kmeans.labels_.tolist()


# In[16]:


reduced['cluster'] = clusters
reduced['name'] = names
reduced.columns = ['x', 'y', 'cluster', 'name']
reduced.head(54)


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


sns.set(style="whitegrid")
ax = sns.lmplot(x="x", y="y", hue='cluster', data = reduced, legend=False,
fit_reg=False, size = 15, scatter_kws={"s": 250})
texts = []
for x, y, s in zip(reduced.x, reduced.y, reduced.name):
    texts.append(plt.text(x, y, s))
ax.set(ylim=(-2, 2))
plt.tick_params(labelsize=15)
plt.xlabel("PC 1", fontsize = 20)
plt.ylabel("PC 2", fontsize = 20)
plt.show()


# In[ ]:




