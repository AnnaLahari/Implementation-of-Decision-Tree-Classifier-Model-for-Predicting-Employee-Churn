#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
data = pd.read_csv("Employee.csv")
data


# In[2]:


data.head()


# In[17]:


data.isnull().sum()


# In[4]:


data["left"].value_counts()


# In[24]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
print('NAME: A.LAHARI')
print('REG.No : 212223230111')


# In[6]:


data["salary"] = le.fit_transform(data["salary"])
data.head()


# In[7]:


x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()


# In[19]:


y=data["left"]
y.head()


# In[9]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 100)


# In[22]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier (criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)


# In[23]:


from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy


# In[13]:


dt.predict([[0.5,0.8,9,260,6,0,1,2]])

