#!/usr/bin/env python
# coding: utf-8

# # Removing fragments from simulation data
# 
# ## Outline
# 
# 1. [Starting point](#starting_point)
# 2. [Data visualization](#visualization)
# 3. [Visualization of the clean data](#clean_data)
# 
# ## Starting point<a id="starting_point"></a>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn as sk
import sys
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    import urllib
    import cloudpickle as cp
else:
    import pickle
    matplotlib.rcParams['figure.dpi'] = 80

print("Pandas version: {}".format(pd.__version__))
print("Numpy version: {}".format(np.__version__))
print("Sklearn version: {}".format(sk.__version__))
print("Running notebook {}".format("in colab." if IN_COLAB else "locally."))

# In[2]:


if not IN_COLAB:
    data_file = "../data/7mm_water_air_plic.pkl"
    with open(data_file, 'rb') as file:
        data = pickle.load(file).drop(["element"], axis=1)
else:
    data_file = "https://github.com/AndreWeiner/machine-learning-applied-to-cfd/blob/master/data/7mm_water_air_plic.pkl?raw=true"
    response = urllib.request.urlopen(data_file)
    data = cp.load(response).drop(["element"], axis=1)

print("The data set contains {} points.".format(data.shape[0]))
data.sample(5)

# ## Data visualization<a id="visualization"></a>

# In[3]:


%matplotlib inline
every = 100
fontsize = 14

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10), sharey=True)

ax1.scatter(data.px.values[::every], data.py.values[::every],
            marker='x', color='C0', s=30, linewidth=0.5)
ax2.scatter(data.pz.values[::every], data.py.values[::every],
            marker='x', color='C0', s=30, linewidth=0.5)

ax1.set_xlabel(r"$x$", fontsize=fontsize)
ax1.set_ylabel(r"$y$", fontsize=fontsize)
ax2.set_xlabel(r"$z$", fontsize=fontsize)

plt.show()

# ## Fragment detection with isolation forests<a id="isolation_forest"></a>

# In[12]:


from sklearn.ensemble import IsolationForest

def fit_isolation_forest(contamination):
    clf = IsolationForest(n_estimators=100, contamination=contamination, behaviour='new')
    return clf.fit(data[['px', 'py', 'pz']].values).predict(data[['px', 'py', 'pz']].values)

# In[13]:


fig, axarr = plt.subplots(1, 6, figsize=(15, 10), sharey=True)

colors = np.array(['C0', 'C1'])

for i, cont in enumerate(np.arange(0.08, 0.135, 0.01)):
    pred = fit_isolation_forest(cont)
    axarr[i].scatter(data.px.values[::every], data.py.values[::every],
                     marker='x', color=colors[(pred[::every] + 1) // 2], s=30, linewidth=0.5)
    axarr[i].set_title("Contamination: {:2.2f}".format(cont))
    axarr[i].set_xlabel(r"$x$", fontsize=fontsize)

axarr[0].set_ylabel(r"$y$", fontsize=fontsize)
plt.show()

# In[14]:


clf = IsolationForest(n_estimators=200, contamination=0.11, behaviour='new')
pred = clf.fit(data[['px', 'py', 'pz']].values).predict(data[['px', 'py', 'pz']].values)

# In[15]:


points_clean = data[['px', 'py', 'pz']].values[pred > 0]
print("Removed {} points from data set.".format(data.shape[0] - points_clean.shape[0]))

# ## Visualization of the clean data<a id="clean_data"></a>

# In[16]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), sharey=True)

ax1.scatter(points_clean[:,0], points_clean[:,1],
            marker='x', color='C1', s=2, linewidth=0.5)
ax2.scatter(points_clean[:,2], points_clean[:,1],
            marker='x', color='C1', s=2, linewidth=0.5)

ax1.set_xlabel(r"$x$", fontsize=fontsize)
ax1.set_ylabel(r"$y$", fontsize=fontsize)
ax2.set_xlabel(r"$z$", fontsize=fontsize)

plt.show()

# In[17]:


%matplotlib inline
# uncomment for interactive plotting with notebook backend   
#%matplotlib notebook
    
every = 50

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_clean[:,0][::every], points_clean[:,2][::every], points_clean[:,1][::every],
            marker='x', color='C1', s=10, linewidth=0.5)

ax.set_xlabel(r"$x$", fontsize=fontsize)
ax.set_ylabel(r"$z$", fontsize=fontsize)
ax.set_zlabel(r"$y$", fontsize=fontsize)
plt.show()

# In[ ]:



