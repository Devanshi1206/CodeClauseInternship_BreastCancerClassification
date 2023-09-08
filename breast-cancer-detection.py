#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Detection

# In[1]:


get_ipython().system('python --version')


# In[2]:


import numpy as np
import pandas as pd 
import os


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore') # To prevent kernel from showing any warning


# **Importing Dataset**

# In[4]:


df = pd.read_csv('data-cancer.csv')


# In[5]:


df.head()


# In[6]:


df.info()


# **Clearing data**

# In[7]:


columns_to_drop = [0,32] 

df.drop(df.columns[columns_to_drop], axis=1, inplace=True)


# In[8]:


# Diagnosis (M = malignant = 1, B = benign = 1)

df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

df['diagnosis'].value_counts()


# In[9]:


df.describe()


# **Data Visualization**

# In[10]:


df.hist(figsize=(15,15))
plt.show()


# In[11]:


plt.figure(figsize=[20,20])
sns.heatmap(df.corr(),annot=True)


# **ML Modeling**

# In[12]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,1:], df['diagnosis'], test_size = 0.2, random_state = 42)


# In[13]:


from sklearn.preprocessing import MinMaxScaler
normal = MinMaxScaler()


# In[14]:


#Fitting Data
normal_fit = normal.fit(x_train)
new_xtrain = normal_fit.transform(x_train)
new_xtest = normal_fit.transform(x_test)
#print(new_xtrain)
#print(new_xtest)


# Using Random Forest Classifier

# In[15]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
rand = RandomForestClassifier()


# In[16]:


#Fitting Data
fit_rand = rand.fit(new_xtrain, y_train)
#predicting score
rand_score = rand.score(new_xtest, y_test)
print('Score of model is : ', rand_score*100,'%')


# **Error Detection**

# In[17]:


#Display Error
#Calculating Mean Squared Error

from sklearn.metrics import mean_squared_error
Yhat = rand.predict(new_xtest)
rand_MSE = mean_squared_error(y_test, Yhat)
rand_RMSE = np.sqrt(rand_MSE)
print('Mean Square Error is: ', rand_MSE)
print('Root Mean Square Error is: ', rand_RMSE)


# **Prediction**

# In[18]:


x_predict = list(rand.predict(x_test))
predicted_df = {'predicted_values': x_predict,'original_values': y_test}
print(classification_report(x_predict, y_test))


# In[19]:


pd.DataFrame(predicted_df).head(10)


# **Accuracy of the model is 96.491% with an error of 0.035%**
