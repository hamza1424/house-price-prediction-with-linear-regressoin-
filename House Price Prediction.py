#!/usr/bin/env python
# coding: utf-8

# In[27]:


# Import Libaries 
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics


# In[28]:


#loading DataFrame
data = pd.read_csv('house.csv')


# In[30]:


# Checking first 10 Records 
data.head(10)


# In[31]:


data.shape


# In[32]:


data.describe()


# In[33]:


# Drawing Plot
data.plot( x = 'SquareFeet', y = "SalePrice", style = "*")
plt.title("Square Feet vs Sale Price")
plt.xlabel("Square Feet")
plt.ylabel("Sale Price")
plt.show()


# In[34]:


# Preparing Data for prediction

X = data.iloc[:,:-1].values
y = data.iloc[:,1].values


# In[35]:


# Train and Test Split


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)


# In[37]:


# Train the model
lr = LinearRegression()
lr.fit(X_train,y_train)


# In[38]:


# Intercept : y = mx+c : m is intercept 
lr.intercept_


# In[39]:


# Co-efficient y = mx+ c : c is co-efficient 
lr.coef_


# In[40]:


# Predicted Values
y_pred = lr.predict(X_test)


# In[48]:


# Actual Vs Predicted Values 
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df.head(20)


# In[ ]:




