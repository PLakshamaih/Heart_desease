#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv(r"C:\Users\Pandilla Lakshmaiah\OneDrive\Desktop\Project_dataset\Heart.csv")


# In[3]:


df


# In[4]:


#deleting column which is not required
df.drop(columns="Unnamed: 0",inplace=True)


# In[5]:


df


# In[7]:


df


# In[8]:


df["ChestPain"]=df["ChestPain"].astype("category")
df["ChestPain"]=df["ChestPain"].cat.codes


# In[9]:


df


# In[10]:


df["ChestPain"]=df["ChestPain"].astype("category")
df["ChestPain"]=df["ChestPain"].cat.codes


# In[11]:


df["AHD"]=df["AHD"].astype("category")
df["AHD"]=df["AHD"].cat.codes
df["Thal"]=df["Thal"].astype("category")
df["Thal"]=df["Thal"].cat.codes


# In[12]:


df


# In[13]:


df.isnull().sum()


# In[15]:


df.dropna(inplace=True)


# In[16]:


df


# In[17]:


X=df.drop(columns="AHD")


# In[18]:


y=df["AHD"]


# In[19]:


from sklearn.model_selection import train_test_split


# In[23]:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=21)


# In[25]:


from sklearn.preprocessing import StandardScaler


# In[26]:


scale=StandardScaler()


# In[27]:


x_train=scale.fit_transform(x_train)
x_test=scale.fit(x_test)


# In[28]:


from sklearn.linear_model import LogisticRegression


# In[29]:


log_reg=LogisticRegression(random_state=0).fit(x_train,y_train)


# In[30]:


log_reg.predict(x_train)


# In[31]:


log_reg.score(x_train,y_train)


# In[ ]:





# In[ ]:




