#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[4]:


pip install lazypredict


# In[6]:


df = pd.read_csv("IRIS.csv")


# In[7]:


def check_data(df,head=5):
    print(20*"-" + "Information".center(20) + 20*"-")
    print(df.info())
    print(20*"-" + "Data Shape".center(20) + 20*"-")
    print(df.shape)
    print("\n" + 20*"-" + "The First 5 Data".center(20) + 20*"-")
    print(df.head())
    print("\n" + 20 * "-" + "The Last 5 Data".center(20) + 20 * "-")
    print(df.tail())
    print("\n" + 20 * "-" + "Missing Values".center(20) + 20 * "-")
    print(df.isnull().sum())
    print("\n" + 40 * "-" + "Describe the Data".center(40) + 40 * "-")
    print(df.describe([0.01, 0.05, 0.10, 0.50, 0.75, 0.90, 0.95, 0.99]).T)
    
check_data(df)


# In[8]:


df['species'].value_counts()


# In[9]:


le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])


# In[10]:


X = df.iloc[ : , :-1] # The features
y = df.iloc[ : , -1] # The Target


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=True)


# In[12]:


LC = LazyClassifier(verbose=1)
models, predictions = LC.fit(X_train, X_test, y_train, y_test)


# In[13]:


predictions


# In[14]:


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression


# In[15]:


lgbm = LGBMClassifier()

lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)

lgbm_T = round(lgbm.score(X_train, y_train) * 100, 2)
lgbm_accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[16]:


knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

knn_T = round(knn.score(X_train, y_train) * 100, 2)
knn_accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[17]:


qda = QuadraticDiscriminantAnalysis()

qda.fit(X_train, y_train)
y_pred = qda.predict(X_test)

qda_T = round(qda.score(X_train, y_train) * 100, 2)
qda_accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[18]:


logreg = LogisticRegression()

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

log = round(logreg.score(X_train, y_train) * 100, 2)
log_accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[19]:


models = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 
              'QuadraticDiscriminantAnalysis', 'LGBMClassifier'],
    
    'Testing_score': [knn_accuracy, log_accuracy,
                       qda_accuracy, lgbm_accuracy],
    
    'Traning_score' : [knn_T, log, qda_T, lgbm_T]})

models.sort_values(by='Testing_score', ascending=False)


# In[ ]:




