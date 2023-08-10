#!/usr/bin/env python
# coding: utf-8

# # <center> Homework-3, Econ 425
# #   <center> Anshika Sharma, UCLA ID:(305488635)

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from getDataset import getDataSet
from sklearn.linear_model import LogisticRegression
from util import Cost_Function, Gradient_Descent, Cost_Function_Derivative, Cost_Function, Prediction, Sigmoid 
import pandas as pd

from pandas import DataFrame
from sklearn import preprocessing
#from sklearn.cross_validation import train_test_split
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel

# Starting codes

# Fill in the codes between "%PLACEHOLDER#start" and "PLACEHOLDER#end"

# step 1: generate dataset that includes both positive and negative samples,
# where each sample is described with two features.
# 250 samples in total.

np.random.seed(425)
[X, y] = getDataSet()  # note that y contains only 1s and 0s,

# create figure for all charts to be placed on so can be viewed together
fig = plt.figure()


def func_DisplayData(dataSamplesX, dataSamplesY, chartNum, titleMessage):
    idx1 = (dataSamplesY == 0).nonzero()  # object indices for the 1st class
    idx2 = (dataSamplesY == 1).nonzero()
    ax = fig.add_subplot(1, 3, chartNum)
    # no more variables are needed
    plt.plot(dataSamplesX[idx1, 0], dataSamplesX[idx1, 1], 'r*')
    plt.plot(dataSamplesX[idx2, 0], dataSamplesX[idx2, 1], 'b*')
    # axis tight
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.set_title(titleMessage)


# plotting all samples
func_DisplayData(X, y, 1, 'All samples')

# number of training samples
nTrain = 120


# In[2]:


######################PLACEHOLDER 1#start#########################
# write you own code to randomly pick up nTrain number of samples for training and use the rest for testing.
# WARNIN: do not use the scikit-learn or other third-party modules for this step

np.random.seed(42)
maxIndex = len(X)
randomTrainingSamples = np.random.choice(maxIndex, nTrain, replace=False)
trainX = X[randomTrainingSamples,:]    #  training samples
trainY = y[randomTrainingSamples,:] # labels of training samples    nTrain X 1

#find indices for data not in the trainins samples
indexes = np.arange(0,250,1)
new = [x for x in indexes if x not in randomTrainingSamples]
 
testX =  X[new,:] # testing samples               
testY = y[new,:]# labels of testing samples     nTest X 1

####################PLACEHOLDER 1#end#########################


# In[3]:


# plot the samples you have pickup for training, check to confirm that both negative and positive samples are included.

# create figure for all charts to be placed on so can be viewed together
fig = plt.figure()

func_DisplayData(X, y, 1, 'All samples')
func_DisplayData(trainX, trainY, 2, 'Training samples')
func_DisplayData(testX, testY, 3, 'Testing samples')

# show all charts
plt.show()
# show all charts
plt.show()


# In[4]:


######################PLACEHOLDER2 #start#########################
# in this placefolder you will need to train a logistic model using the training data: trainX, and trainY.
# please delete these coding lines and use the sample codes provided in the folder "codeLogit" -- done :)


########################################################################
################# training and testing using sklearn    #########
########################################################################
clf = LogisticRegression()
# call the function fit() to train the class instance
clf.fit(trainX,trainY)

# visualize data using functions in the library pylab
pos = where(y == 1)
neg = where(y == 0)
scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
xlabel('Feature 1: score 1')
ylabel('Feature 2: score 2')
legend(['Label: Admitted', 'Label: Not Admitted'])
show()



########################################################################
##############Step: training and testing using self-developed model ##
########################################################################

##Gradient descent method to learn logistic regression
theta = [0,0] #initial model parameters
alpha = 0.1 # learning rates
max_iteration = 1000 # maximal iterations


m = len(y) # number of samples

for x in range(max_iteration):
        # call the functions for gradient descent method
        new_theta = Gradient_Descent(X,y,theta,m,alpha)
        theta = new_theta
        if x % 200 == 0:
            # calculate the cost function with the present theta
            Cost_Function(X,y,theta,m)
            print('theta ', theta)
            print('cost is ', Cost_Function(X,y,theta,m))
            

########################################################################            
#################         Step-5: comparing two models         #########
########################################################################
#Accuracies of 2 models.             
            
#get score for sklearn method
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX,testY)
length = len(testX)
for i in range(length):
        prediction = round(Prediction(testX[i],theta))
        answer = testY[i]
        if prediction == answer:
            score += 1

#get score for gradient descent method and compare to sk learn
my_score = float(score) / float(length)
if my_score > scikit_score:
        print('You won!')
elif my_score == scikit_score:
        print('Its a tie!')
else:
        print('Scikit won.. :(')
print('Your score: ', my_score)
print('Scikits score: ', scikit_score) 
######################PLACEHOLDER2 #end #########################


# In[5]:


# step 3: Use the model to get class labels of testing samples.

######################PLACEHOLDER3 #start#########################
# codes for making prediction, 
# with the learned model, apply the logistic model over testing samples
# hatProb is the probability of belonging to the class 1.
# y = 1/(1+exp(-Xb))
# yHat = 1./(1+exp( -[ones( size(X,1),1 ), X] * bHat )); ));
# WARNING: please DELETE THE FOLLOWING CODEING LINES and write your own codes for making predictions

#prediction with gradient descent method

predGD = [Prediction(i,theta) for i in testX]
predGD = [float(int(j>=.5)) for j in predGD]

#prediction with sklearn

predSK = clf.predict(testX)

print("GD method: ", predGD)
print("sklearn method: ", predSK)
######################PLACEHOLDER 3 #end #########################


# In[6]:


# step 4: evaluation
# compare predictions and true labels to calculate average error and standard deviation

testY = testY.reshape(-1)
testYDiff = np.abs(predGD - testY)
avgErr = np.mean(testYDiff)
stdErr = np.std(testYDiff)

testYDiff1 = np.abs(predSK - testY)
avgErr1 = np.mean(testYDiff1)
stdErr1 = np.std(testYDiff1)

print('GD average error: {} ({})'.format(avgErr, stdErr))

print('SK average error: {} ({})'.format(avgErr1, stdErr1))


# In[7]:


######################PLACEHOLDER4 #start#########################
#Please add codes to calculate the ROC curves and AUC of the trained model over all the testing samples
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

######### Roc and Auc for Self developed model #####
length = len(testX)
prop4 =[]
pred4 = []
for i in range(length):
    prop4 = Prediction(testX[i],theta)
    pred4.append(prop4)
    
pred_prob4 = np.array(pred4)
fpr4, tpr4, thresh4 = roc_curve(testY, pred_prob4, pos_label=1)
#print('Score of GD without biased term:', my_score)
print('The AUC score is: ', roc_auc_score(testY, pred_prob4))


# In[8]:


random_probs = [0 for i in range(len(testY))]
p_fpr, p_tpr, _ = roc_curve(testY, random_probs, pos_label=1)
# plot roc curves
plt.plot(fpr4, tpr4, linestyle='--',color='green', 
         label='GD - Model without biased-term')
#plt.plot(fpr2, tpr2, linestyle='--',color='green', label='My own model')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve of model From method-2, GD, without biased-term')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')
#plt.legend(loc='best')
#plt.savefig('ROC',dpi=300)
plt.show()


# In[9]:


######### Roc and Auc for Scikit model #####

# Scikit learning
pred_prob1 = clf.predict_proba(testX)
# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(testY, pred_prob1[:,1], pos_label=1)
#plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='pink', 
         label='Using Scikit-Learning')
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
#probs = [0 for _ in range(len(testY))]
auc = roc_auc_score(testY, pred_prob1[:,1])
print('The AUC score is: ', roc_auc_score(testY, pred_prob1[:,1]))
######################PLACEHOLDER4 #end #########################


# #### <center> Part-2

# In[ ]:


#Question 1 in the pdf


# In[10]:


#Question 2 confusion matrix

#function should return accuracy, per class precision, per class recall rate
#for a 2 by two matrix with ground truth labels 0 and 1

 
def func_calConfusionMatrix(predY,trueY):
    uniqueVals = [0,1]
    matrix=np.zeros((2,3))
    for j in uniqueVals :
        #for value 0
        trueVal = 0
        falseVal = 0
        
        for i in range(len(predY)):
            if(trueY[i]==j):
                if(int(predY[i])==int(j)):
                    trueVal = trueVal+1
                else:
                    falseVal = falseVal+1
        if(j==0):
            matrix[j]=[j,trueVal,falseVal]
        else:
            matrix[j]=[j,falseVal,trueVal]
        
    accuracy = (matrix[0,1]+matrix[1,2])/len(predY)
    prec_0 = matrix[0,1]/(matrix[0,1]+matrix[1,1])
    prec_1 = matrix[1,2]/(matrix[1,2]+matrix[0,2])
    recall_0 = matrix[0,1]/(matrix[0,1]+matrix[0,2])
    recall_1 = matrix[1,2]/(matrix[1,1]+matrix[1,2])
    print("Confusion Matrix:")
    print(matrix)
    print('Accuracy:',accuracy)
    print('Precision 0:', prec_0)
    print('Precision 1:', prec_1)
    print('Recall 0:', recall_0)
    print('Recall 1:', recall_1)
 

print("SK Learn Confusion Matrix:")
func_calConfusionMatrix(predSK,testY)

        
print("GD Confusion Matrix:")
func_calConfusionMatrix(predGD,testY)


# ### <center> Part-3

# ### <center> Cross Validation

# ### 1. $\alpha$=0.1.

# In[14]:



from sklearn.model_selection import KFold
# K fold, k=5
kf = KFold(n_splits=5)
score1=[]
for train_index, test_index in kf.split(X):
#    split train set and test set
    train_X_c, train_y_c = X[train_index], y[train_index]
    test_X_c, test_y_c = X[test_index], y[test_index]
    #set hyperparameters for self-implemented model
    theta = [0,0] #initial model parameters
    alpha = 0.1 # learning rates
    max_iteration = 1000 # maximal iterations
    m = len(train_y_c) # number of samples
    score=0
    #fit the model
    for x in range(max_iteration):
        # call the functions for gradient descent method
        new_theta = Gradient_Descent(train_X_c,train_y_c,theta,m,alpha)
        theta = new_theta
        if x % 200 == 0:
            # calculate the cost function with the present theta
            Cost_Function(train_X_c,train_y_c,theta,m)
    #Get accuracy score of the model
    for i in range(len(test_y_c)):
        prediction = round(Prediction(test_X_c[i],theta))
        answer = test_y_c[i]
        if prediction == answer:
            score += 1

    my_score = float(score) / float(len(test_y_c))
    #print and store score time of each cross validation
    print(my_score)
    score1.append(my_score)


# In[15]:


#Get the mean accuracy score
np.mean(score1)


# The mean accuracy of the self-implemented model when $\alpha=0.1$ is 0.768.
# 
# ### 2. $\alpha$=0.001.

# In[16]:


# K fold, k=5
kf = KFold(n_splits=5)
score2=[]
for train_index, test_index in kf.split(X):
#    split train set and test set
    train_X_c, train_y_c = X[train_index], y[train_index]
    test_X_c, test_y_c = X[test_index], y[test_index]
    #set hyperparameters for self-implemented model
    theta = [0,0] #initial model parameters
    alpha = 0.001 # learning rates
    max_iteration = 1000 # maximal iterations
    m = len(train_y_c) # number of samples
    score=0
    #fit the model
    for x in range(max_iteration):
        # call the functions for gradient descent method
        new_theta = Gradient_Descent(train_X_c,train_y_c,theta,m,alpha)
        theta = new_theta
        if x % 200 == 0:
            # calculate the cost function with the present theta
            Cost_Function(train_X_c,train_y_c,theta,m)
    #Get accuracy score of the model
    for i in range(len(test_y_c)):
        prediction = round(Prediction(test_X_c[i],theta))
        answer = test_y_c[i]
        if prediction == answer:
            score += 1

    my_score = float(score) / float(len(test_y_c))
    #print and store score time of each cross validation
    print(my_score)
    score2.append(my_score)


# In[17]:


#Get the mean accuracy score
np.mean(score2)


# The mean accuracy of the self-implemented model when  $\alpha=0.001$  is 0.776.
# 
# ### 3. $\alpha$=0.002

# In[18]:


# K fold, k=5
kf = KFold(n_splits=5)
score3=[]
for train_index, test_index in kf.split(X):
#    split train set and test set
    train_X_c, train_y_c = X[train_index], y[train_index]
    test_X_c, test_y_c = X[test_index], y[test_index]
    #set hyperparameters for self-implemented model
    theta = [0,0] #initial model parameters
    alpha = 0.002 # learning rates
    max_iteration = 1000 # maximal iterations
    m = len(train_y_c) # number of samples
    score=0
    #fit the model
    for x in range(max_iteration):
        # call the functions for gradient descent method
        new_theta = Gradient_Descent(train_X_c,train_y_c,theta,m,alpha)
        theta = new_theta
        if x % 200 == 0:
            # calculate the cost function with the present theta
            Cost_Function(train_X_c,train_y_c,theta,m)
    #Get accuracy score of the model
    for i in range(len(test_y_c)):
        prediction = round(Prediction(test_X_c[i],theta))
        answer = test_y_c[i]
        if prediction == answer:
            score += 1

    my_score = float(score) / float(len(test_y_c))
    #print and store score time of each cross validation
    print(my_score)
    score3.append(my_score)


# In[19]:


#Get the mean accuracy score
np.mean(score3)


# The mean accuracy of the self-implemented model when $\alpha=0.002$ is 0.776.

# ### 4. $\alpha$=0.005

# In[20]:


# K fold, k=5
kf = KFold(n_splits=5)
score4=[]
for train_index, test_index in kf.split(X):
#    split train set and test set
    train_X_c, train_y_c = X[train_index], y[train_index]
    test_X_c, test_y_c = X[test_index], y[test_index]
    #set hyperparameters for self-implemented model
    theta = [0,0] #initial model parameters
    alpha = 0.005 # learning rates
    max_iteration = 1000 # maximal iterations
    m = len(train_y_c) # number of samples
    score=0
    #fit the model
    for x in range(max_iteration):
        # call the functions for gradient descent method
        new_theta = Gradient_Descent(train_X_c,train_y_c,theta,m,alpha)
        theta = new_theta
        if x % 200 == 0:
            # calculate the cost function with the present theta
            Cost_Function(train_X_c,train_y_c,theta,m)
    #Get accuracy score of the model
    for i in range(len(test_y_c)):
        prediction = round(Prediction(test_X_c[i],theta))
        answer = test_y_c[i]
        if prediction == answer:
            score += 1

    my_score = float(score) / float(len(test_y_c))
    #print and store score time of each cross validation
    print(my_score)
    score4.append(my_score)


# In[21]:


#Get the mean accuracy score
np.mean(score4)


# The mean accuracy of the self-implemented model when $\alpha=0.005$ is 0.78.

# ### 5. $\alpha$=0.01

# In[22]:


# K fold, k=5
kf = KFold(n_splits=5)
score5=[]
for train_index, test_index in kf.split(X):
#    split train set and test set
    train_X_c, train_y_c = X[train_index], y[train_index]
    test_X_c, test_y_c = X[test_index], y[test_index]
    #set hyperparameters for self-implemented model
    theta = [0,0] #initial model parameters
    alpha = 0.01 # learning rates
    max_iteration = 1000 # maximal iterations
    m = len(train_y_c) # number of samples
    score=0
    #fit the model
    for x in range(max_iteration):
        # call the functions for gradient descent method
        new_theta = Gradient_Descent(train_X_c,train_y_c,theta,m,alpha)
        theta = new_theta
        if x % 200 == 0:
            # calculate the cost function with the present theta
            Cost_Function(train_X_c,train_y_c,theta,m)
    #Get accuracy score of the model
    for i in range(len(test_y_c)):
        prediction = round(Prediction(test_X_c[i],theta))
        answer = test_y_c[i]
        if prediction == answer:
            score += 1

    my_score = float(score) / float(len(test_y_c))
    #print and store score time of each cross validation
    print(my_score)
    score5.append(my_score)


# In[23]:


#Get the mean accuracy score
np.mean(score5)


# The mean accuracy of the self-implemented model when $\alpha=0.01$ is 0.784.

# ### 6. $\alpha=0.02$

# In[24]:


# K fold, k=5
kf = KFold(n_splits=5)
score6=[]
for train_index, test_index in kf.split(X):
#    split train set and test set
    train_X_c, train_y_c = X[train_index], y[train_index]
    test_X_c, test_y_c = X[test_index], y[test_index]
    #set hyperparameters for self-implemented model
    theta = [0,0] #initial model parameters
    alpha = 0.02 # learning rates
    max_iteration = 1000 # maximal iterations
    m = len(train_y_c) # number of samples
    score=0
    #fit the model
    for x in range(max_iteration):
        # call the functions for gradient descent method
        new_theta = Gradient_Descent(train_X_c,train_y_c,theta,m,alpha)
        theta = new_theta
        if x % 200 == 0:
            # calculate the cost function with the present theta
            Cost_Function(train_X_c,train_y_c,theta,m)
    #Get accuracy score of the model
    for i in range(len(test_y_c)):
        prediction = round(Prediction(test_X_c[i],theta))
        answer = test_y_c[i]
        if prediction == answer:
            score += 1

    my_score = float(score) / float(len(test_y_c))
    #print and store score time of each cross validation
    print(my_score)
    score6.append(my_score)


# In[25]:


#Get the mean accuracy score
np.mean(score6)


# The mean accuracy of the self-implemented model when $\alpha=0.02$ is 0.772.

# ### 7. $\alpha$=0.05

# In[26]:


# K fold, k=5
kf = KFold(n_splits=5)
score7=[]
for train_index, test_index in kf.split(X):
#    split train set and test set
    train_X_c, train_y_c = X[train_index], y[train_index]
    test_X_c, test_y_c = X[test_index], y[test_index]
    #set hyperparameters for self-implemented model
    theta = [0,0] #initial model parameters
    alpha = 0.05 # learning rates
    max_iteration = 1000 # maximal iterations
    m = len(train_y_c) # number of samples
    score=0
    #fit the model
    for x in range(max_iteration):
        # call the functions for gradient descent method
        new_theta = Gradient_Descent(train_X_c,train_y_c,theta,m,alpha)
        theta = new_theta
        if x % 200 == 0:
            # calculate the cost function with the present theta
            Cost_Function(train_X_c,train_y_c,theta,m)
    #Get accuracy score of the model
    for i in range(len(test_y_c)):
        prediction = round(Prediction(test_X_c[i],theta))
        answer = test_y_c[i]
        if prediction == answer:
            score += 1

    my_score = float(score) / float(len(test_y_c))
    #print and store score time of each cross validation
    print(my_score)
    score7.append(my_score)


# In[27]:


#Get the mean accuracy score
np.mean(score7)


# ### 8. $\alpha$=0.07

# In[28]:


# K fold, k=5
kf = KFold(n_splits=5)
score8=[]
for train_index, test_index in kf.split(X):
#    split train set and test set
    train_X_c, train_y_c = X[train_index], y[train_index]
    test_X_c, test_y_c = X[test_index], y[test_index]
    #set hyperparameters for self-implemented model
    theta = [0,0] #initial model parameters
    alpha = 0.07 # learning rates
    max_iteration = 1000 # maximal iterations
    m = len(train_y_c) # number of samples
    score=0
    #fit the model
    for x in range(max_iteration):
        # call the functions for gradient descent method
        new_theta = Gradient_Descent(train_X_c,train_y_c,theta,m,alpha)
        theta = new_theta
        if x % 200 == 0:
            # calculate the cost function with the present theta
            Cost_Function(train_X_c,train_y_c,theta,m)
    #Get accuracy score of the model
    for i in range(len(test_y_c)):
        prediction = round(Prediction(test_X_c[i],theta))
        answer = test_y_c[i]
        if prediction == answer:
            score += 1

    my_score = float(score) / float(len(test_y_c))
    #print and store score time of each cross validation
    print(my_score)
    score8.append(my_score)


# In[29]:


#Get the mean accuracy score
np.mean(score8)


# The mean accuracy of the self-implemented model when $\alpha=0.07$ is 0.768.

# ### 9. $\alpha$=0.5

# In[30]:


# K fold, k=5
kf = KFold(n_splits=5)
score9=[]
for train_index, test_index in kf.split(X):
#    split train set and test set
    train_X_c, train_y_c = X[train_index], y[train_index]
    test_X_c, test_y_c = X[test_index], y[test_index]
    #set hyperparameters for self-implemented model
    theta = [0,0] #initial model parameters
    alpha = 0.5 # learning rates
    max_iteration = 1000 # maximal iterations
    m = len(train_y_c) # number of samples
    score=0
    #fit the model
    for x in range(max_iteration):
        # call the functions for gradient descent method
        new_theta = Gradient_Descent(train_X_c,train_y_c,theta,m,alpha)
        theta = new_theta
        if x % 200 == 0:
            # calculate the cost function with the present theta
            Cost_Function(train_X_c,train_y_c,theta,m)
    #Get accuracy score of the model
    for i in range(len(test_y_c)):
        prediction = round(Prediction(test_X_c[i],theta))
        answer = test_y_c[i]
        if prediction == answer:
            score += 1

    my_score = float(score) / float(len(test_y_c))
    #print and store score time of each cross validation
    print(my_score)
    score9.append(my_score)


# In[31]:


#Get the mean accuracy score
np.mean(score9)


# The mean accuracy of the self-implemented model when $\alpha=0.5$ is 0.768.

# ### 10. $\alpha$=1

# In[32]:


# K fold, k=5
kf = KFold(n_splits=5)
score10=[]
for train_index, test_index in kf.split(X):
#    split train set and test set
    train_X_c, train_y_c = X[train_index], y[train_index]
    test_X_c, test_y_c = X[test_index], y[test_index]
    #set hyperparameters for self-implemented model
    theta = [0,0] #initial model parameters
    alpha = 1 # learning rates
    max_iteration = 1000 # maximal iterations
    m = len(train_y_c) # number of samples
    score=0
    #fit the model
    for x in range(max_iteration):
        # call the functions for gradient descent method
        new_theta = Gradient_Descent(train_X_c,train_y_c,theta,m,alpha)
        theta = new_theta
        if x % 200 == 0:
            # calculate the cost function with the present theta
            Cost_Function(train_X_c,train_y_c,theta,m)
    #Get accuracy score of the model
    for i in range(len(test_y_c)):
        prediction = round(Prediction(test_X_c[i],theta))
        answer = test_y_c[i]
        if prediction == answer:
            score += 1

    my_score = float(score) / float(len(test_y_c))
    #print and store score time of each cross validation
    print(my_score)
    score10.append(my_score)


# In[45]:


#Get the mean accuracy score
np.mean(score10)


# The mean accuracy of the self-implemented model when $\alpha=1$ is 0.768.

# |alpha|0.001|0.002|0.005|0.01|0.02|0.05|0.07|0.1|0.5|1.0|
# |-----|-----|-|-|-|-|-|-|-|-|-|
# Accuracy|0.768|0.776|0.776|0.78|0.784|0.772|0.768|0.768|0.768|0.768|
# 
# The optimal learning rate is 0.02, with accuracy 0.784. But it is still less than the sklearn model. 

# ### <center> Training model with data (training samples)

# In[11]:


#Preapare dataframe for showing the result
alphar = [0.001, 0.002, 0.005, 0.01, 0.02,0.05, 0.07, 0.1,0.5, 1.0]
col = ['iteration']+ ['testing_score'] + ['coef_x_%d'%i for i in range(0,2)]
ind = ['alpha_%.2g'%alphar[i] for i in range(0,10)]
coef_matrix_logist = pd.DataFrame(index=ind, columns=col)


# In[12]:


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


# In[13]:


thetat = np.zeros(2) 
max_iterationr = [1000] # sets of max-iteration
total =  []
m = trainX.shape[0]
for j in range (1):
#Iterate through all powers and assimilate results
    for i in range(10):
        coef_matrix_logist.iloc[i,0] = max_iterationr[j]
        coef_matrix_logist.iloc[i,1:] = logisticreg(trainX , trainY,thetat,m,alphar[i],testX,testY,max_iterationr[j])
    print(coef_matrix_logist)


# the model with the optimal learning rate does not outperform all other candidates. 
