#!/usr/bin/env python
# coding: utf-8

# In[67]:


#import the necessary libraries
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import *
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from util import func_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import sklearn.svm as svm

#Load data
df = pd.read_excel("425data.xls", sheet_name='Sheet1')
df.info(verbose=True)

#format data
Date = pd.to_datetime(list(df['Date']))
Recession = list(df['Recession'])
CPI = list(df['CPI'])
Unemployment = list(df['Unemployment'])
Production = list(df['Production'])
WorkHour = list(df['WorkHour'])
HousingStart = list(df['HousingStart'])
PersonalIncome = list(df['PersonalIncome'])
LongTermYield = list(df['LongTermYield'])
BusinessConfidence = list(df['BusinessConfidence'])

# Description for Recession
plt.bar(["0","1"], height = [Recession.count(0), Recession.count(1)])


# In[68]:


# OLS regression to choose variables for logistic regression
X_ols = df.iloc[:,2:10]
Y_ols = df.iloc[:,1]

# add a column for constant term
X_ols = sm.add_constant(X_ols)

# fitting OLS
model = sm.OLS(Y_ols, X_ols)
result = model.fit()
print(result.summary())


# In[69]:


# Aggregate the data chosen by the OLS; not include CPI, PersonalIncome, or LongTermYield
aggregateList = [Unemployment, Production, WorkHour, HousingStart, BusinessConfidence]
aggregateArray = np.array(aggregateList, dtype='float')
X = np.transpose(aggregateArray)
Y = np.array(Recession)

#Split the data to train and test
nMax = 707
nTrain = round(nMax * 2/3)
seed(0)
rand()

trainIndex = np.random.choice(nMax, nTrain, replace=False)
testIndex = np.setdiff1d(range(nMax), trainIndex)

trainX = X[trainIndex]
trainY = Y[trainIndex]
testX = X[testIndex]
testY = Y[testIndex]


# In[80]:


# Logistic regression
clf = LogisticRegression(solver='lbfgs', random_state=0)
clf.fit(trainX, trainY)

prediction = clf.predict(testX)

plt.title('Logistic Predictions')
plt.plot(prediction, label='prediction')
plt.plot(testY, label='real')
plt.legend(loc='upper right')
plt.xlabel('Time Index')
plt.ylabel('Recession')
plt.show()


# In[71]:


# Results
cm, acc, arrR, arrP = func_confusion_matrix(testY, prediction)
print(cm)
print(acc)
print(arrR)
print(arrP)


# In[81]:


# K-Nearest Neighbor
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(trainX, trainY)
knn_pred = knn.predict(testX)

plt.title('KNN Predictions')
plt.plot(knn_pred, label='prediction')
plt.plot(testY, label='real')
plt.xlabel('Time Index')
plt.ylabel('Recession')
plt.legend(loc='upper right')
plt.show()

cm, acc, arrR, arrP = func_confusion_matrix(testY, knn_pred)
print(cm)
print(acc)
print(arrR)
print(arrP)


# In[73]:


# Support Vector Machine
# Choose the best c_value
c_range = range(1,10)
svm_c_error = []
for c_value in c_range:
    svm_model = svm.SVC(kernel='linear', C=c_value, gamma='scale')
    svm_model.fit(X=trainX, y=trainY)
    error = 1. - svm_model.score(testX, testY)
    svm_c_error.append(error)
plt.plot(c_range, svm_c_error)
plt.title('Linear SVM')
plt.xlabel('c values')
plt.ylabel('error')


# In[74]:


# Choose the best kernel
kernel_types = ['linear', 'poly', 'rbf']
svm_kernel_error = []
for kernel_value in kernel_types:
    svm_model = svm.SVC(kernel=kernel_value, C=4, gamma='scale')
    svm_model.fit(X=trainX, y=trainY)
    error = 1. - svm_model.score(testX, testY)
    svm_kernel_error.append(error)
plt.plot(kernel_types, svm_kernel_error)
plt.title('SVM by Kernels')
plt.xlabel('Kernel')
plt.ylabel('error')
plt.xticks(kernel_types)
plt.show()


# In[75]:


# Best model for SVM
best_kernel = 'linear'
best_c = 4
svm_model = svm.SVC(kernel=best_kernel, C=best_c)
svm_model.fit(trainX, trainY)


# In[82]:


# Prediction
svm_pred = svm_model.predict(testX)

plt.title('Linear SVM Predictions')
plt.plot(svm_pred, label='prediction')
plt.plot(testY, label='real')
plt.xlabel('Time Index')
plt.ylabel('Recession')
plt.legend(loc = 'upper right')
plt.show()


# In[77]:


cm, acc, arrR, arrP = func_confusion_matrix(testY, svm_pred)
print(cm)
print(acc)
print(arrR)
print(arrP)

