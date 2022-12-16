#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("WineQT.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[6]:


df.describe().T


# In[5]:


#describes the dataset

df.describe()


# In[6]:


df.describe().T

#transposes index & columns of the table


# In[7]:


df.isnull()

#checks whether there are NULL values


# In[8]:


df.isnull().sum()

#sum of NULL values


# <b> if there were any missing data:
# 
# for col in df.columns:
#     
#     if df[col].isnull().sum() > 0:
#     
#         df[col] = df[col].fillna(df[col].mean())
# 
# df.isnull().sum()

# # Histogram
# To visualise distribution of data with continuous values in columns of dataset.

# In[9]:


df.hist(bins = 20, figsize = (10,10))
plt.show()


# # COUNT PLOT
# To visualise number data of each quality of wine

# In[10]:


plt.bar(df['quality'], df['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()


# # HEATMAP
# To remove redundant features

# In[11]:


plt.figure(figsize=(12,12))
sb.heatmap(df.corr() > 0.7, annot = True, cbar=False)
plt.show()


# # MODEL DEVELOPMENT
# Let’s prepare our data for training and splitting it into training and validation data so, that we can select which model’s performance is best as per the use case. We will train some of the state of the art machine learning classification models and then select best out of them using validation data.

# In[12]:


df['best quality'] = [1 if x > 5 else 0 for x in df.quality]


# We replace the column with 'object' data type with '0 and 1' as there are only two categories!

# In[13]:


df.replace({'white': 1, 'red': 0}, inplace=True)


# After segregating features & the target variable from the dataset, 
# we split in into 80:20 for MODEL SELECTION.

# In[14]:


features = df.drop(['quality', 'best quality'], axis =1)
target = df['best quality']

xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size = 0.2, random_state=40)

xtrain.shape, xtest.shape


# # NORMALISING THE DATA
# Normalising the data before training helps us to achieve stable and fast training of the model.

# In[15]:


norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)


# As the data has been prepared completely,
# let's train some state of the art machine learning model on it.

# In[16]:


models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]

for i in range(3):
    models[i].fit(xtrain, ytrain)
    
    print(f'{models[i]}: ')
    print('Training Accuracy: ', metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
    print('Validation Accuracy: ', metrics.roc_auc_score(ytest, models[i].predict(xtest)))
    print()


# From the above accuarcies, we can say that
# 
#    <b> Logistic Regression and SVC performed better than XGBClassifier
#     
# on the validation data with less difference between the validation and training data.

# # CONFUSION MATRIX
# Plot confusion matrix for validation data using logistic regression model:

# In[17]:


metrics.plot_confusion_matrix(models[0], xtest, ytest)
plt.show()


# Plot confusion matrix for validation data using XGBClassifier model:

# In[18]:


metrics.plot_confusion_matrix(models[1], xtest, ytest)
plt.show()


# Plot confusion matrix for validation data using SVC model:

# In[19]:


metrics.plot_confusion_matrix(models[2], xtest, ytest)
plt.show()


# # CLASSIFICATION REPORT

# Print classification report of linear regression model:

# In[20]:


print(metrics.classification_report(ytest, models[0].predict(xtest)))


# Print classification report of XGBCLassifier:

# In[21]:


print(metrics.classification_report(ytest, models[1].predict(xtest)))


# Print classification report of SVC model:

# In[22]:


print(metrics.classification_report(ytest, models[2].predict(xtest)))


# In[ ]:





# Dataset used: Wine Quality Dataset, Kaggle
# (https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)

# Authored by:
#     Soumya Kushwaha
