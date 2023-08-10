#!/usr/bin/env python
# coding: utf-8

# #  <center> Quiz-2 <br>
# #  <center>  Anshika Sharma, UCLA ID:(305488635) 

# In[1]:


#In the first step, we import all the functions:
import math
import numpy as np
import pandas as pd

from pandas import DataFrame
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel

#Now, we import self-defined functions:
from util import Cost_Function, Gradient_Descent, Cost_Function_Derivative, Cost_Function, Prediction, Sigmoid


# Step: pre-processing the data

# In[2]:


# scale data to be between -1,1 

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
df = pd.read_csv("data.csv", header=0)

# clean up data
df.columns = ["grade1","grade2","label"]

x = df["label"].map(lambda x: float(x.rstrip(';')))

# formats the input data into two arrays, one of independant variables
# and one of the dependant variable
X = df[["grade1","grade2"]]
X = np.array(X)
X = min_max_scaler.fit_transform(X)
Y = df["label"].map(lambda x: float(x.rstrip(';')))
Y = np.array(Y)

print(X.shape)
print(Y.shape)


# Splitting the data

# In[3]:


# split the dataset into two subsets: testing and training
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33, random_state = 0)


# Step: training and testing using sklearn

# In[4]:


# use sklearn class
clf = LogisticRegression()
# call the function fit() to train the class instance
clf.fit(X_train,Y_train)
# scores over testing samples
print(clf.score(X_test,Y_test))

# visualize data using functions in the library pylab 
pos = where(Y == 1)
neg = where(Y == 0)
scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
xlabel('Feature 1: score 1')
ylabel('Feature 2: score 2')
legend(['Label:  Admitted', 'Label: Not Admitted'])
show()


# Step-4: training and testing using self-developed model Without bias term

# In[5]:


#Without bias term
thetao = [0,0] #initial model parameters
alpha = 0.1 # learning rates
max_iteration = 1000 # maximal iterations

m = len(Y_test) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X_test,Y_test,thetao,m,alpha)
	thetao = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X_test,Y_test,thetao,m)
		print('theta ', thetao)
		print('cost is ', Cost_Function(X_test,Y_test,thetao,m))
print("coefficeint of model without biase term: ", thetao)
 


# Step: training and testing using self-developed model with biased term

# In[6]:


thetab = [0,0,0] #initial model parameters
alphab = 0.1  # learning rates
max_iteration = 1000 # maximal iterations

xValuesb = np.ones((len(Y_train) , 3)) #create array(60,3) of 1 
xValuesb[:, 1:3] = X_train[:, 0:2] # split training and testing data set 
yValuesb = Y_train

m = len(Y_train) # number of samples
total_c = []
for x in range(max_iteration):
    # call the functions for gradient descent method
    new_theta = Gradient_Descent(xValuesb ,yValuesb,thetab,m,alphab)
    thetab = new_theta
    cost = Cost_Function(xValuesb,yValuesb,thetab,m)
    total_c.append(cost)
    if x % 200 == 0:
        # calculate the cost function with the present theta
        print('theta ', thetab)
        print('cost is ', cost)
res = [thetab]
print('theta of final model:',res)
print('cost:', cost)
import matplotlib.pyplot as plt


# In[7]:


scoreo = 0
scoreb = 0

# accuracy for sklearn
scikit_score = clf.score(X_test,Y_test)
length = len(X_test)
#model without bias term
for i in range(length):
	predictiono = round(Prediction(X_test[i],thetao))
	answero = Y_test[i]
	if predictiono == answero:
		scoreo += 1
my_scoreo = float(scoreo) / float(length) 
        
for i in range(length):
	predictionb = round(Prediction(X_test[i],thetab))
	answero = Y_test[i]
	if predictionb == answero:
		scoreb += 1
my_scoreb = float(scoreb) / float(length)

print('The score of Scikit model: ', scikit_score)
print('The score of model without biased term : ', my_scoreo)
print('The score of model with biased term: ', my_scoreb)     


# ### <center> Changing hyperparameters, learning rate and max_iteration

# ### <center> 1 - Self developed model without biased term (Developed using Method-2 of Logistic reg)

# In[8]:


##### training and testing using self-developed model without biased term #####

# Change alpha and iteration

def logisticreg(xValues ,yValues,theta,m,alpha,testXValues,Y_test,max_iteration):
    for x in range(max_iteration):
    # call the functions for gradient descent method
        new_theta = Gradient_Descent(xValues ,yValues,theta,m,alpha)
        theta = new_theta
        Cost_Function(xValues,yValues,theta,m)  
    #evaluate model
    score = 0
    length = len(testXValues)
    for i in range(length):
        prediction = round(Prediction(testXValues[i],theta))
        answer = Y_test[i]
        if prediction == answer:
            score += 1
    my_score = float(score) / float(length)
    res = [my_score]
    res.extend(theta)
    return res

alphar = [0.0001, 0.0007,0.001, 0.007,0.01, 0.07,0.1,0.7 ]
#Initialize the dataframe to store coefficients
col = ['max_iteration']+ ['testing_score'] + ['coef_x_%d'%i for i in range(1,3)]
ind = ['alpha_%.2g'%alphar[i] for i in range(0,8)]
#ind = a+a+a
coef_matrix_logisw = pd.DataFrame(index=ind, columns=col)

thetaw = [0,0] #initial model parameters
xValuesw = X_train # split training and testing data set 
yValuesw = Y_train

m = len(Y_train) # number of samples

testXValues2 = X_test
max_iterationr = [500,1000,10000]
total =  []
for j in range (3):
#Iterate through all powers and assimilate results
    for i in range(8):
        coef_matrix_logisw.iloc[i,0] = max_iterationr[j]
        coef_matrix_logisw.iloc[i,1:] = logisticreg(xValuesw ,yValuesw,thetaw,m,alphar[i],testXValues2,Y_test,max_iterationr[j])
    print(coef_matrix_logisw)   


# ### <center> 1.1 - My self developed best model without biased term (Developed using Method-2 of Logistic reg)

# In[9]:


#plotting the convergence curve

#######training and testing using self-developed model biased term (best model)##

thetaw = [0 , 0] #initial model parameters
alphaw = 0.001  # learning rates
max_iteration = 500 # maximal iterations

xValuesw = X_train # split training and testing data set 
yValuesw = Y_train

m = len(Y_train) # number of samples
total_cw = []
for x in range(max_iteration):
    # call the functions for gradient descent method
    new_thetaw = Gradient_Descent(xValuesw ,yValuesw,thetaw,m,alphaw)
    thetaw = new_thetaw
    costw = Cost_Function(xValuesw,yValuesw,thetaw,m)
    total_cw.append(costw)
    if x % 200 == 0:
        # calculate the cost function with the present theta
        print('theta ', thetaw)
        print('cost is ', costw)
resw = [thetaw]
print('theta of final model:',resw)
print('cost:', costw)
import matplotlib.pyplot as plt

plt.plot(range(0,len(total_cw)),total_cw);
plt.xlabel('iteration')
plt.ylabel('cost')
plt.title('alpha = {}  theta = {}'.format(alphaw, thetaw))
plt.show()


# In[10]:


# ROC, AUC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
pred4 = []
pred4 = []
Ytest4 = []
thetaob =   thetaw
print(thetaob)
length = len(X_test)
score = 0   
for i in range(length):
	prediction = round(Prediction(X_test[i],thetaob))
	answer = Y_test[i]
	if prediction == answer:
		score += 1
my_scoreob = float(score) / float(length)

prop4 =[]
for i in range(length):
    prop4 = Prediction(X_test[i],thetaob)
    pred4.append(prop4)

pred_prob4 = np.array(pred4)
fpr4, tpr4, thresh4 = roc_curve(Y_test, pred_prob4, pos_label=1)
print('Score of my best model without biased term:', my_scoreob)
print('The AUC score is: ', roc_auc_score(Y_test, pred_prob4))

random_probs = [0 for i in range(len(Y_test))]
p_fpr, p_tpr, _ = roc_curve(Y_test, random_probs, pos_label=1)
    


# In[11]:


# plot roc curves
plt.plot(fpr4, tpr4, linestyle='--',color='green', label='my own-model without biased-term')
#plt.plot(fpr2, tpr2, linestyle='--',color='green', label='My own model')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve of My best model (From method-2) without biased-term')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

#plt.legend(loc='best')
#plt.savefig('ROC',dpi=300)
plt.show()


# ### <center> 2 - My Self developed model with biased term (Developed using Method-2 of Logistic reg)

# In[12]:



##############training and testing using self-developed model biased term#############

# Change iteration and learning rate, alpha

def logisticreg(xValuesl ,yValuesl,thetal,m,alpha,testXValues1,Y_test,max_iteration):
    for x in range(max_iteration):
    # call the functions for gradient descent method
        new_theta = Gradient_Descent(xValuesl ,yValuesl,thetal,m,alpha)
        thetal = new_theta
        Cost_Function(xValuesl,yValuesl,thetal,m)  
    #evaluate model
    scorel = 0
    length = len(testXValues1)
    for i in range(length):
        predictionl = round(Prediction(testXValues1[i],thetal))
        answerl = Y_test[i]
        if predictionl == answerl:
            scorel += 1
    my_scorel = float(scorel) / float(length)
    res = [my_scorel]
    res.extend(thetal)
    return res

alphar = [0.0001, 0.0007,0.001, 0.007,0.01, 0.07,0.1,0.7 ]
#Initialize the dataframe to store coefficients
col = ['max_iteration']+ ['testing_score'] + ['coef_x_%d'%i for i in range(0,3)]
ind = ['alpha_%.2g'%alphar[i] for i in range(0,8)]
#ind = a+a+a
coef_matrix_logis = pd.DataFrame(index=ind, columns=col)

thetal = [0,0,0] #initial model parameters
xValuesl = np.ones((len(Y_train) , 3)) #create array(60,3) of 1 
xValuesl[:, 1:3] = X_train[:, 0:2] # split training and testing data set 
yValuesl = Y_train

m = len(Y_train) # number of samples

testXValues1 = np.ones((len(X_test), 3)) 
testXValues1[:, 1:3] = X_test[:, 0:2]
max_iterationr = [500,1000,10000]
total =  []
for j in range (3):
#Iterate through all powers and assimilate results
    for i in range(8):
        #print(logisticreg(xValuesl ,yValuesl,thetal,m,alphar[i],testXValues1,Y_test,max_iteration))
        coef_matrix_logis.iloc[i,0] = max_iterationr[j]
        coef_matrix_logis.iloc[i,1:] = logisticreg(xValuesl ,yValuesl,thetal,m,alphar[i],testXValues1,Y_test,max_iterationr[j])
    print(coef_matrix_logis)


# ### <center> 2.1 - My Self Developed best model without biased term

# In[16]:



##############training and testing using self-developed model biased term (best model)#########

thetab = [0,0,0] #initial model parameters
alphab = 0.007  # learning rates
max_iteration = 500 # maximal iterations

xValuesb = np.ones((len(Y_train) , 3)) #create array(60,3) of 1 
xValuesb[:, 1:3] = X_train[:, 0:2] # split training and testing data set 
yValuesb = Y_train

m = len(Y_train) # number of samples
total_c = []
for x in range(max_iteration):
    # call the functions for gradient descent method
    new_theta = Gradient_Descent(xValuesb ,yValuesb,thetab,m,alphab)
    thetab = new_theta
    cost = Cost_Function(xValuesb,yValuesb,thetab,m)
    total_c.append(cost)
    if x % 200 == 0:
        # calculate the cost function with the present theta
        print('theta ', thetab)
        print('cost is ', cost)
reso = [thetab]
print('theta of final model:',reso)
print('cost:', cost)
import matplotlib.pyplot as plt


# In[17]:


#visualize the convergence curve
plt.plot(range(0,len(total_c)),total_c);
plt.xlabel('iteration')
plt.ylabel('cost')
plt.title('alpha = {}  theta = {}'.format(alpha, thetab))
plt.show()


# In[18]:


# ROC, AUC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
pred1 = []
pred1 = []
Ytest1 = []
thetaob = thetab  
print(thetaob)
length = len(X_test)
scoreb = 0   

# My best model
testXValues1 = np.ones((len(X_test), 3)) 
testXValues1[:, 1:3] = X_test[:, 0:2]
for i in range(length):
	predictionb = round(Prediction(testXValues1[i],thetaob))
	answerb = Y_test[i]
	if predictionb == answerb:
		scoreb += 1
        
my_scoreob = float(scoreb) / float(length)

for i in range(length):
    prop1 = Prediction(testXValues1[i],thetaob)
    pred1.append(prop1)
pred_prob2 = np.array(pred1)
fpr2, tpr2, thresh2 = roc_curve(Y_test, pred_prob2, pos_label=1)
print('Score of my best model with biased term : ', my_scoreob)
print('AUC score', roc_auc_score(Y_test, pred_prob2))
    


# In[19]:


# plot roc curves
#plt.plot(fpr4, tpr4, linestyle='--',color='orange', label='Using Scikit-Learning')
plt.plot(fpr2, tpr2, linestyle='--',color='purple', label='My own model')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve of My best model (Method-2) with biased-term')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

#plt.legend(loc='best')
#plt.savefig('ROC',dpi=300)
plt.show()


# In[20]:


# Scikit learning

pred_prob1 = clf.predict_proba(X_test)
# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(Y_test, pred_prob1[:,1], pos_label=1)

#plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='pink', label='Using Scikit-Learning')
#plt.plot(fpr2, tpr2, linestyle='--',color='green', label='My own model')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve of Scikit learning model')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

#plt.legend(loc='best')
#plt.savefig('ROC',dpi=300)
plt.show()
print("Scikit-learn, accuracy score:", clf.score(X_test, Y_test))
print("Scikit-learn, auc score:", roc_auc_score(Y_test, pred_prob1[:,1]))

# In[21]:


Y_test.mean()


# In[22]:


Y_train.mean()

