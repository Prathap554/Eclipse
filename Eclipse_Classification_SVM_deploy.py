#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Importing the Libraries

import numpy as np
import pandas as pd
import pickle

from sklearn.svm import SVC


# In[2]:


## Reading the dataset

df = pd.read_csv("Consolidated_SVM_Cleaned.csv")

df.head()


# In[3]:


X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#X_new = df.drop('Size', axis=1)
#y_new = df['Size']


# In[4]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X = sc.fit_transform(X)


# ## Training the SVM model on train dataset

# In[5]:


classifier = SVC(kernel = 'rbf', gamma = 0.1, C=1000, random_state = 0)
classifier.fit(X, y)

# In[7]:

# Saving model to disk
pickle.dump(classifier, open('SVM_model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('SVM_model.pkl','rb'))
print(model.predict([[0.738726199,8.765929222,3.4854645730000002,4.266696453,3.922311306,2.136468649]]))


# In[ ]:




