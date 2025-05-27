#!/usr/bin/env python
# coding: utf-8

# In[103]:


import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random as rd


# In[104]:


genes=['gene' + str(i) for i in range(1,101)]
wt=['wt' + str(i) for i in range(1,6)]
ko=['ko' + str(i) for i in range(1,6)]

data=pd.DataFrame(columns=[*wt,*ko], index=genes)
for gene in data.index:
   data.loc[gene,'wt1':'wt5']=np.random.poisson(lam=rd.randrange(10,1000),size=5)
   data.loc[gene,'ko1':'ko5']=np.random.poisson(lam=rd.randrange(10,1000),size=5)
print(data.head())
data.shape


# In[105]:


scaler=StandardScaler()
scaled_data=scaler.fit_transform(data)
data_scaled=pd.DataFrame(scaled_data,index=data.index, columns=data.columns)
print(data_scaled.head())


# In[106]:


pca=PCA()
pca.fit(data_scaled)
pca_data=pca.transform(data_scaled)


# In[107]:


per_var=np.round(pca.explained_variance_ratio_*100,decimals=1)
labels=['pc'+str(i) for i in range(1,len(per_var)+1)]


plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance') 
plt.xlabel('Principal Component') 
plt.title('Scree Plot') 
plt.show() 


# In[119]:




# Assume pca_data and per_var are already defined
labels = ['pc' + str(i) for i in range(1, len(per_var) + 1)]
index_labels = ['sample_' + str(i) for i in range(pca_data.shape[0])]

# Create DataFrame
pca_df = pd.DataFrame(pca_data, index=index_labels, columns=labels)

# Scatter plot of first two PCs
plt.figure(figsize=(8, 6))
plt.scatter(pca_df.pc1, pca_df.pc2)
plt.xlabel(f'PC1 - {per_var[0]}% Explained Variance')
plt.ylabel(f'PC2 - {per_var[1]}% Explained Variance')
plt.title('Scatter Plot of PC1 vs PC2')
plt.grid(True)
plt.show()

# Scree plot
plt.figure(figsize=(8, 6))
plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()


# In[ ]:





# In[ ]:




