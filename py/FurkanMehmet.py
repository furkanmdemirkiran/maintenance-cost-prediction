#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


# In[2]:


df = pd.read_excel("GrupC.xlsx")
df


# In[3]:


df = df.drop("Bakim_ID", axis = 1)
df


# In[4]:


df ["Bakim_Onceligi"].value_counts()


# In[5]:


df[df["Bakim_Onceligi"] == "?"]


# In[6]:


df ["Bakim_Onceligi"] = df["Bakim_Onceligi"].replace("?", pd.NA)
df ["Bakim_Onceligi"].value_counts()


# In[7]:


df["Vardiya"].value_counts()


# In[8]:


df["Tesis_Bolgesi"].value_counts()


# In[9]:


df ["Tesis_Bolgesi"] = df["Tesis_Bolgesi"].replace("Batı", "Bati")
df ["Tesis_Bolgesi"].value_counts()


# In[10]:


df["Makine_Saglik_Durumu"].value_counts()


# In[11]:


df.isnull().sum() #sutün kategorik ise mod | #sayısal değerler ise medyan ve mod


# In[12]:


x = df.groupby(["Tesis_Bolgesi", "Makine_Saglik_Durumu"])["Bakim_Onceligi"].transform(
    lambda x: x.mode()[0]

)
df["Bakim_Onceligi"] = df["Bakim_Onceligi"].fillna(x)
df["Bakim_Onceligi"].isnull().sum()


# In[13]:


x = df.groupby(["Tesis_Bolgesi", "Makine_Saglik_Durumu"])["Vardiya"].transform(
    lambda x: x.mode()[0]

)
df["Vardiya"] = df["Vardiya"].fillna(x)
df["Vardiya"].isnull().sum()


# In[14]:


x = df.groupby(["Vardiya", "Makine_Saglik_Durumu"])["Tesis_Bolgesi"].transform(
    lambda x: x.mode()[0]

)
df["Tesis_Bolgesi"] = df["Tesis_Bolgesi"].fillna(x)
df["Tesis_Bolgesi"].isnull().sum()


# In[15]:


x = df.groupby(["Vardiya"])["Tesis_Bolgesi"].transform(
    lambda x: x.mode()[0]

)
df["Tesis_Bolgesi"] = df["Tesis_Bolgesi"].fillna(x)
df["Tesis_Bolgesi"].isnull().sum()


# In[16]:


x = df.groupby(["Vardiya", "Tesis_Bolgesi"])["Makine_Saglik_Durumu"].transform(
    lambda x: x.mode()[0]

)
df["Makine_Saglik_Durumu"] = df["Makine_Saglik_Durumu"].fillna(x)
df["Makine_Saglik_Durumu"].isnull().sum()


# In[17]:


x = df.groupby(["Vardiya", "Tesis_Bolgesi"])["Makine_Yasi"].transform("median")
df["Makine_Yasi"] = df["Makine_Yasi"].fillna(x)
df["Makine_Yasi"].isnull().sum()


# In[18]:


x = df.groupby(["Vardiya", "Tesis_Bolgesi"])["Calisma_Saati"].transform("median")
df["Calisma_Saati"] = df["Calisma_Saati"].fillna(x)
df["Calisma_Saati"].isnull().sum()


# In[19]:


x = df.groupby(["Vardiya", "Tesis_Bolgesi"])["Titreşim_Seviyesi"].transform("mean")
df["Titreşim_Seviyesi"] = df["Titreşim_Seviyesi"].fillna(x)
df["Titreşim_Seviyesi"].isnull().sum()


# In[20]:


x = df.groupby(["Vardiya", "Tesis_Bolgesi"])["Ariza_Sayisi"].transform("median")
df["Ariza_Sayisi"] = df["Ariza_Sayisi"].fillna(x)
df["Ariza_Sayisi"].isnull().sum()


# In[21]:


x = df.groupby(["Vardiya", "Tesis_Bolgesi"])["Duruş_Suresi_Saat"].transform("median")
df["Duruş_Suresi_Saat"] = df["Ariza_Sayisi"].fillna(x)
df["Duruş_Suresi_Saat"].isnull().sum()


# In[22]:


x = df.groupby(["Vardiya", "Tesis_Bolgesi"])["Ortam_Sicakligi"].transform("median")
df["Ortam_Sicakligi"] = df["Ortam_Sicakligi"].fillna(x)
df["Ortam_Sicakligi"].isnull().sum()


# In[23]:


df.isnull().sum()


# In[24]:


df


# In[25]:


df= pd.get_dummies(df, columns=["Vardiya","Tesis_Bolgesi"], drop_first=True, dtype=int)
df


# In[26]:


df ["Bakim_Onceligi"].value_counts()


# In[27]:


df["Bakim_Onceligi"] = df ["Bakim_Onceligi"].map({
    "Kritik": 0,
    "Dusuk": 1,
    "Orta": 2,
    "Yuksek":3
})

df["Makine_Saglik_Durumu"] = df ["Makine_Saglik_Durumu"].map({
    "Kritik": 0,
    "Yıpranmış": 1,
    "Zayıf": 2,
    "İyi": 3,
    "Kusursuz":4
})  


# In[28]:


df


# In[38]:


df = df.dropna()


# In[39]:


X = df.drop("Toplam_Bakim_Maliyeti", axis = 1)
Y = df["Toplam_Bakim_Maliyeti"]


# In[48]:


x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.2,
                                                    random_state=42) #Verinin bölünme işlemini gerçekleştiriyorum
skaler = StandardScaler()
sutunlar = ["Bakim_Onceligi", "Makine_Saglik_Durumu", "Makine_Yasi", "Calisma_Saati",
           "Titreşim_Seviyesi", "Ariza_Sayisi", "Duruş_Suresi_Saat", "Ortam_Sicakligi"]

x_train[sutunlar] = skaler.fit_transform(x_train[sutunlar])
x_test[sutunlar] = skaler.transform(x_test[sutunlar])


# In[41]:


model = LinearRegression()
model.fit(x_train, y_train)


# In[45]:


y_pred = model.predict(x_test)


# In[46]:


mse = mean_squared_error(y_pred, y_test)
r2 = r2_score(y_pred, y_test)
mse, r2


# In[ ]:




