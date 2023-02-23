#!/usr/bin/env python
# coding: utf-8

# In[150]:


import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# In[151]:


CovidDataSet= pd.read_csv("D:/All Project Files/Covid 19 Determination/Covid Dataset.csv")
pd.pandas.set_option('display.max_columns',None)

CovidDataSet


# In[153]:


CovidDataSet.info()


# In[154]:


CovidDataSet.describe(include='all')


# In[155]:


CovidDataSet.columns


# In[156]:


CovidDataSet.isnull().sum()


# In[157]:


sns.countplot(x='COVID-19',data=CovidDataSet)


# In[158]:


CovidDataSet["COVID-19"].value_counts().plot.pie(explode=[0.1,0.1],autopct='%1.1f%%',shadow=True)
plt.title('Number of Positive Cases');


# In[159]:


sns.countplot(x='Breathing Problem',hue='COVID-19',data=CovidDataSet)
plt.title('Breathing Problem Presence in Positive & Negative Cases');


# In[160]:


sns.countplot(x='Fever',hue='COVID-19',data=CovidDataSet);
plt.title('Fever Presence in Positive & Negative Cases');


# In[161]:


sns.countplot(x='Dry Cough',hue='COVID-19',data=CovidDataSet)
plt.title('Dry Cough Presence in Positive & Negative Cases');


# In[162]:


sns.countplot(x='Sore throat',hue='COVID-19',data=CovidDataSet)
plt.title('Sore throat Presence in Positive & Negative Cases');


# In[163]:


sns.countplot(x='Abroad travel',hue='COVID-19',data=CovidDataSet)
plt.title('Abroad Travel Occurence in Positive & Negative Cases');


# In[164]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder() 

#converting true and false value to 1 and 0


# In[167]:


CovidDataSet['Breathing Problem']=le.fit_transform(CovidDataSet['Breathing Problem'])
CovidDataSet['Fever']=le.fit_transform(CovidDataSet['Fever'])
CovidDataSet['Dry Cough']=le.fit_transform(CovidDataSet['Dry Cough'])
CovidDataSet['Sore throat']=le.fit_transform(CovidDataSet['Sore throat'])
CovidDataSet['Running Nose']=le.fit_transform(CovidDataSet['Running Nose'])
CovidDataSet['Asthma']=le.fit_transform(CovidDataSet['Asthma'])
CovidDataSet['Chronic Lung Disease']=le.fit_transform(CovidDataSet['Chronic Lung Disease'])
CovidDataSet['Headache']=le.fit_transform(CovidDataSet['Headache'])
CovidDataSet['Heart Disease']=le.fit_transform(CovidDataSet['Heart Disease'])
CovidDataSet['Diabetes']=le.fit_transform(CovidDataSet['Diabetes'])
CovidDataSet['Hyper Tension']=le.fit_transform(CovidDataSet['Hyper Tension'])
CovidDataSet['Abroad travel']=le.fit_transform(CovidDataSet['Abroad travel'])
CovidDataSet['Contact with COVID Patient']=le.fit_transform(CovidDataSet['Contact with COVID Patient'])
CovidDataSet['Attended Large Gathering']=le.fit_transform(CovidDataSet['Attended Large Gathering'])
CovidDataSet['Visited Public Exposed Places']=le.fit_transform(CovidDataSet['Visited Public Exposed Places'])
CovidDataSet['Family working in Public Exposed Places']=le.fit_transform(CovidDataSet['Family working in Public Exposed Places'])
CovidDataSet['Wearing Masks']=le.fit_transform(CovidDataSet['Wearing Masks'])
CovidDataSet['Sanitization from Market']=le.fit_transform(CovidDataSet['Sanitization from Market'])
CovidDataSet['COVID-19']=le.fit_transform(CovidDataSet['COVID-19'])
CovidDataSet['Dry Cough']=le.fit_transform(CovidDataSet['Dry Cough'])
CovidDataSet['Sore throat']=le.fit_transform(CovidDataSet['Sore throat'])
CovidDataSet['Gastrointestinal ']=le.fit_transform(CovidDataSet['Gastrointestinal '])
CovidDataSet['Fatigue ']=le.fit_transform(CovidDataSet['Fatigue '])


# In[168]:


CovidDataSet


# In[169]:


CovidDataSet.dtypes.value_counts()


# In[170]:


CovidDataSet.describe(include='all')


# In[173]:


corr = CovidDataSet.corr()
plt.figure(figsize = (30,25))
sns.heatmap(corr, annot = True, cmap = plt.cm.Reds)


# In[174]:


CovidDataSet=CovidDataSet.drop(['Chronic Lung Disease','Heart Disease','Gastrointestinal ','Wearing Masks','Sanitization from Market','Asthma','Diabetes'], axis=1)


# In[175]:


CovidDataSet.columns


# In[176]:


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score


# In[177]:


x=CovidDataSet.drop('COVID-19',axis=1)
y=CovidDataSet['COVID-19']

CovidDataSet=CovidDataSet.drop('COVID-19',axis=1)


# In[178]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)


# In[220]:


from sklearn import tree

# Max Depth None
DT = tree.DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=42)

DT.fit(x_train,y_train)

y_pred = DT.predict(x_test)

#Score/Accuracy

acc_decisiontree=DT.score(x_test, y_test)*100

acc_decisiontree


# In[221]:


Target = ['Negative','Positive']

import graphviz
# DOT data
DTRep = tree.export_graphviz(DT, out_file=None, 
                                feature_names= CovidDataSet.columns,
                                class_names= Target,
                                filled=True)

# Draw graph
DTgraphmax = graphviz.Source(DTRep, format="png") 
DTgraphmax


# In[222]:


# Max Depth 5
DT = tree.DecisionTreeClassifier(max_depth=5, min_samples_split=2,random_state=42)

DT.fit(x_train,y_train)

y_pred = DT.predict(x_test)

#Score/Accuracy

acc_decisiontree=DT.score(x_test, y_test)*100

acc_decisiontree


# In[223]:


DTRep = tree.export_graphviz(DT, out_file=None, 
                                feature_names= CovidDataSet.columns,
                                class_names= Target,
                                filled=True)

# Draw graph
DTgraph5 = graphviz.Source(DTRep, format="png") 
DTgraph5


# In[224]:


# Max Depth 7
DT = tree.DecisionTreeClassifier(max_depth=7, min_samples_split=2,random_state=42)

DT.fit(x_train,y_train)

y_pred = DT.predict(x_test)

#Score/Accuracy

acc_decisiontree=DT.score(x_test, y_test)*100

acc_decisiontree


# In[225]:


DTRep = tree.export_graphviz(DT, out_file=None, 
                                feature_names= CovidDataSet.columns,
                                class_names= Target,
                                filled=True)

# Draw graph
DTgraph7 = graphviz.Source(DTRep, format="png") 
DTgraph7


# In[229]:


DTgraphmax.render("decision_tree_MaxDepth_None")
DTgraph7.render("decision_tree_MaxDepth_7")
DTgraph5.render("decision_tree_MaxDepth_5")


# In[230]:


# Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=1000)
#Fit

model.fit(x_train, y_train)
#Score/Accuracy
acc_randomforest=model.score(x_test, y_test)*100
acc_randomforest


# In[231]:


# KNN

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
#Score/Accuracy
acc_knn=knn.score(x_test, y_test)*100
acc_knn


# In[234]:


# Naive Bayes

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train,y_train)
#Score/Accuracy
acc_gaussian= model.score(x_test, y_test)*100
acc_gaussian


# In[235]:


# Support Vector Model

from sklearn import svm
#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel
#Train the model using the training sets
clf.fit(x_train, y_train)
#Predict the response for test dataset
y_pred = clf.predict(x_test)
#Score/Accuracy
acc_svc=clf.score(x_test, y_test)*100
acc_svc


# In[236]:


# Logistic Regression

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
#Fit the model
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
#Score/Accuracy
acc_logreg=model.score(x_test, y_test)*100
acc_logreg


# Summary of Results Accuracy
# 
# Decision Trees
# Max depth none            - 98.22194972409565 (Highest Accuracy)
# Max depth five            - 95.70815450643777
# Max depth seven           - 96.56652360515021
# 
# Random Forest Regressor   - 92.38816126191503
# KNN                       - 96.7504598405886
# Naive Bayes               - 76.5787860208461 (Lowest Accuracy)
# Support Vector Model      - 96.32127529123238
# Logistic Regression       - 96.7504598405886
