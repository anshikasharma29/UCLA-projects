#!/usr/bin/env python
# coding: utf-8

# ### <center> Econ 425 Homework Assignment 1, Part-II
# ### <center> Anshika Sharma, UCLA ID(305488635)

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression
import numpy as np
from download_data import download_data #from the download_data.py file, I imported the function "download_data"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from GD import gradientDescent
from dataNormalization import rescaleMatrix #from the datanormalization.py file, I imported these function.
from dataNormalization import rescaleNormalization
from dataNormalization import meanNormalization


# In[2]:


from sklearn.preprocessing import MinMaxScaler
#load the dataset
sat = download_data('sat.csv', [1, 2, 4]).values
sat1 = sat.copy()
# Normalize data by using MinMaxScaler
sat = MinMaxScaler().fit_transform(sat)

#
print("Data shape: {}".format(sat.shape))
#splitting the data set in the same way as that of part-1
X_train = sat[0:60, 0:2]
X_test = sat[60:len(sat),0:2]
y_train = sat[0:60, 2]
y_test = sat[60:len(sat),2] 


# ### 1. Linear regresssion using Scikit Learn

# In[3]:


# linear regression model using Scikit Learn
from pandas import DataFrame
lr = LinearRegression().fit(X_train, y_train) # fitting the training and testing data set in the lr model
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))
resl = ["linear regession"] # creating a list "resl" and adding training, testing score and the intercepts and coefficients to it.
resl.extend([lr.score(X_train, y_train)])
resl.extend([lr.score(X_test, y_test)])
resl.extend([lr.intercept_])
resl.extend(lr.coef_)  
resl

#creating a data frame/matrix with all the elemnts of the list "resl"
linear = pd.DataFrame(columns =['Model', 'training_score', 'testing_score', 'intercept', 'coef_x_1', 'coef_x_2'])
linear.loc[0,:] = resl
linear


# ### 2. Ridge regression using Scikit learn

# In[4]:


# ridge regression AKA linear regression with l2 regularization
from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))


# In[5]:


# creating the function, "ridge_regression"
def ridge_regression(X_train, X_test, y_train, y_test, alpha, max_iter ):
    #Fit the model
    ridgereg = Ridge(alpha=alpha, max_iter= max_iter)
    ridgereg.fit(X_train, y_train)
    #Return the result in pre-defined format 
    resr = [ridgereg.score(X_train, y_train)]
    resr.extend([ridgereg.score(X_test, y_test)])
    resr.extend([ridgereg.intercept_])
    resr.extend(ridgereg.coef_)  
    return resr


# In[6]:


#Define the alpha values to test
#Note that we are trying out the different values of the hyper-parameter, here it being alpha and save the 
#various alpha parameters in a list and name it "alphar"
alphar = [1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 0.1,1, 5, 10,15]

#Initialize the dataframe to store coefficients
col = ['training_score','testing_score','intercept'] + ['coef_x_%d'%i for i in range(1,3)]
ind = ['alpha_%.2g'%alphar[i] for i in range(0,11)]
coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)

#Iterate through all powers and assimilate results
for i in range(11):
    coef_matrix_ridge.iloc[i,] = ridge_regression(X_train, X_test, y_train, y_test, alphar[i], 10000)


# In[7]:


#The following code gives the matrix
pd.options.display.float_format = '{:,.5g}'.format
coef_matrix_ridge


# In[8]:


coef_matrix_ridge['testing_score'].max() #extracting the highest testing score


# In[9]:


bestridge = coef_matrix_ridge.loc[coef_matrix_ridge.testing_score == coef_matrix_ridge['testing_score'].max()] 
bestridge.insert(0, 'Model', "Ridge_model")
bestridge
#Note that bestridge extracts that row from "coef_matrix_ridge" in which the testing score is maximum and I save it as "bestridge"


# ### 3. Lasso regression using Scikit learn

# In[10]:


##### Lasso Method AKA linear regression with L1 regularization

from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used:", np.sum(lasso.coef_ != 0))


# In[11]:


# creating the function, "lasso_regression"
def lasso_regression(X_train, X_test, y_train, y_test, alpha, max_iter):
    #Fit the model
    lassoreg = Lasso(alpha=alpha, max_iter= max_iter)
    lassoreg.fit(X_train, y_train)
    #Return the result in pre-defined format 
    resl = [lassoreg.score(X_train, y_train)]
    resl.extend([lassoreg.score(X_test, y_test)])
    resl.extend([lassoreg.intercept_])
    resl.extend(lassoreg.coef_)  
    return resl


# In[12]:


#Define the alpha values to test
#Note that we are modifying the values of the hyper-parameter, here it being alpha and save the 
#various alpha parameters in a list and name it "alphar"
alphar = [1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 0.1,1, 5, 10,15]

#Initialize the dataframe to store coefficients
col = ['training_score','testing_score','intercept'] + ['coef_x_%d'%i for i in range(1,3)]
ind = ['alpha_%.2g'%alphar[i] for i in range(0,11)]
coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

#Iterate through all powers and assimilate results
for i in range(11):
    coef_matrix_lasso.iloc[i,] = lasso_regression(X_train, X_test, y_train, y_test, alphar[i], 10000)


# In[13]:


#The following code gives out the matrix
pd.options.display.float_format = '{:,.4g}'.format
coef_matrix_lasso


# In[14]:


coef_matrix_lasso['testing_score'].max() #extracting the highest testing score


# In[15]:


bestlasso = coef_matrix_lasso.loc[coef_matrix_lasso.testing_score == coef_matrix_lasso['testing_score'].max()] 
bestlasso.insert(0, 'Model', "Lasso_model")
bestlasso
#Note that bestlasso extracts that row from "coef_matrix_lasso" in which the testing score is maximum and I save it as "bestlasso"


# ### 4. Elastic regression using Scikit Learn

# In[16]:


from sklearn.linear_model import ElasticNet


# In[17]:


# creating the function, "elastic_regression"
def elastic_regression(X_train, X_test, y_train, y_test, alpha, max_iter ):
    #Fit the model
    elasticreg = ElasticNet(alpha=alpha, max_iter= max_iter)
    elasticreg.fit(X_train, y_train)
    #Return the result in pre-defined format 
    rese = [elasticreg.score(X_train, y_train)]
    rese.extend([elasticreg.score(X_test, y_test)])
    rese.extend([elasticreg.intercept_])
    rese.extend(elasticreg.coef_)  
    return rese


# In[18]:


# l1_ratio=0.5

#Note that we are modifying the values of the hyper-parameter, here it being alpha and learning ratio. We save the 
#various alpha parameters in a list and name it "alphar". As far as the learning rate is concerned, we define it in the function 
#itself, once taking it as 0.5 and once taking it as 0.3 and 0.1

#Define the alpha values to test
alphar =[1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 0.1,1, 5, 10,15]

#Initialize the dataframe to store coefficients
col = ['training_score','testing_score','intercept'] + ['coef_x_%d'%i for i in range(1,3)]
ind = ['alpha_%.2g'%alphar[i] for i in range(0,11)]
coef_matrix_elastic = pd.DataFrame(index=ind, columns=col)

#Iterate through all powers and assimilate results
for i in range(11):
    coef_matrix_elastic.iloc[i,] = elastic_regression(X_train, X_test, y_train, y_test, alphar[i], 10000)


# In[19]:


#This code gives out the matrix 
pd.options.display.float_format = '{:,.4g}'.format
coef_matrix_elastic


# In[20]:


bestelastic = coef_matrix_elastic.loc[coef_matrix_elastic.testing_score == coef_matrix_elastic['testing_score'].max()] #extracting the highest testing score
bestelastic.insert(0, 'Model', "Elastic_model")
bestelastic 
#Note that bestelastic extracts that row from "coef_matrix_elastic" in which the testing score is maximum and I save it as "bestelastic"


# In[21]:


# l1_ratio=0.3
def elastic_regression(X_train, X_test, y_train, y_test, alpha, max_iter):
    #Fit the model
    elasticreg = ElasticNet(alpha=alpha, max_iter= max_iter,  l1_ratio=0.3)
    elasticreg.fit(X_train, y_train)
    #Return the result in pre-defined format 
    rese = [elasticreg.score(X_train, y_train)]
    rese.extend([elasticreg.score(X_test, y_test)])
    rese.extend([elasticreg.intercept_])
    rese.extend(elasticreg.coef_)  
    return rese
#Define the alpha values to test
alphar = [1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 0.1,1, 5, 10,15]

#Initialize the dataframe to store coefficients
col = ['training_score','testing_score','intercept'] + ['coef_x_%d'%i for i in range(1,3)]
ind = ['alpha_%.2g'%alphar[i] for i in range(0,11)]
coef_matrix_elastic = pd.DataFrame(index=ind, columns=col)

#Iterate through all powers and assimilate results
for i in range(11):
    coef_matrix_elastic.iloc[i,] = elastic_regression(X_train, X_test, y_train, y_test, alphar[i], 10000)
    
pd.options.display.float_format = '{:,.5g}'.format
coef_matrix_elastic   


# In[22]:


bestelastic = coef_matrix_elastic.loc[coef_matrix_elastic.testing_score == coef_matrix_elastic['testing_score'].max()] 
bestelastic.insert(0, 'Model', "Elastic_model")
bestelastic


# In[23]:


# l1_ratio=0.1
def elastic_regression(X_train, X_test, y_train, y_test, alpha, max_iter):
    #Fit the model
    elasticreg = ElasticNet(alpha=alpha, max_iter= max_iter,  l1_ratio=0.05)
    elasticreg.fit(X_train, y_train)
    #Return the result in pre-defined format 
    rese = [elasticreg.score(X_train, y_train)]
    rese.extend([elasticreg.score(X_test, y_test)])
    rese.extend([elasticreg.intercept_])
    rese.extend(elasticreg.coef_)  
    return rese

#Define the alpha values to test
alphar =[1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 0.1,1, 5, 10,15]

#Initialize the dataframe to store coefficients
col = ['training_score','testing_score','intercept'] + ['coef_x_%d'%i for i in range(1,3)]
ind = ['alpha_%.2g'%alphar[i] for i in range(0,11)]
coef_matrix_elastic = pd.DataFrame(index=ind, columns=col)

#Iterate through all powers and assimilate results
for i in range(11):
    coef_matrix_elastic.iloc[i,] = elastic_regression(X_train, X_test, y_train, y_test, alphar[i], 10000)
    
pd.options.display.float_format = '{:,.5g}'.format
coef_matrix_elastic   


# In[24]:


bestelastic = coef_matrix_elastic.loc[coef_matrix_elastic.testing_score == coef_matrix_elastic['testing_score'].max()] 
bestelastic.insert(0, 'Model', "Elastic_model")
bestelastic


# ### 5. Linear Regression function using gradient descent function aka lr_scratch

# In[25]:


from download_data import download_data
import numpy as np
import matplotlib.pyplot as plt
#from GD import gradientDescent
from dataNormalization import rescaleMatrix
from dataNormalization import rescaleNormalization
from dataNormalization import meanNormalization


# In[26]:


import numpy as np
# X          - single array/vector
# y          - single array/vector
# theta      - single array/vector
# alpha      - scalar
# iterations - scarlar

def gradientDescent(X, y, theta, alpha, numIterations):
    '''
    # This function returns a tuple (theta, Cost array)
    '''
    m = len(y)
    arrCost =[];
    transposedX = np.transpose(X) # transpose X into a vector  -> XColCount X m matrix
    for interation in range(0, numIterations):
        ################PLACEHOLDER4 #start##########################
        #: write your codes to update theta, i.e., the parameters to estimate. 
        # Replace the following variables if needed 
        residualError =   X.dot(theta)-y #res error = diff b/w predicted and true values = (true value or y) - (theta*x)
        gradient =  transposedX.dot(residualError)/m #gradient = 1/m(transpose x)*(res error) --> transpose for matrix mult.
        change = [alpha * x for x in gradient] #think of change as new - old theta
        theta = np.subtract(theta, change)  # or theta = theta - alpha * gradient

        ################PLACEHOLDER4 #end##########################
        
        ################PLACEHOLDER5 #start##########################
        # calculate the current cost with the new theta; 
        atmp  = (1 / 2*m) * np.sum(residualError ** 2)  #calculate the cost function
        print(atmp)
        arrCost.append(atmp) 
        # cost = (1 / m) * np.sum(residualError ** 2)
     
        ################PLACEHOLDER5 #end##########################
        
    return theta, arrCost


# In[27]:


# d
#Define the alpha values to test
theta = np.zeros(3)
lr_rate = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 0.1, 0.5, 0.7,1] # here this represents the learning rate 
MAX_ITER = 10000 #maximum iterations
#Initialize the dataframe to store coefficients
col = ['training_score','testing_score','intercept'] + ['coef_x_%d'%i for i in range(1,3)]
ind = ['learning_rate_%.2g'%lr_rate[i] for i in range(0,11)]
coef_matrix_grad =pd.DataFrame(index=ind, columns=col)

#Iterate through all powers and assimilate results
xValues = np.ones((len(X_train), 3))
xValues[:, 1:3] = X_train[:,:]
yValues = y_train

testXValues = np.ones((len(y_test), 3)) 
testXValues[:, 1:3] = X_test[:, 0:2]

from sklearn.metrics import r2_score
for i in range(11):
    [theta, arrCost] = gradientDescent(xValues, yValues, theta, lr_rate[i], MAX_ITER)
    train_r2 = r2_score(yValues,xValues.dot(theta))
    test_r2=  r2_score(y_test,testXValues.dot(theta))
    a = [train_r2]
    a.extend([test_r2])
    a.extend(theta)  
    coef_matrix_grad.iloc[i,] = a


# In[28]:


#this code generates the matrix
pd.options.display.float_format = '{:,.4g}'.format
coef_matrix_grad


# In[29]:


bestlr_scratch = coef_matrix_grad.loc[coef_matrix_grad.testing_score == coef_matrix_grad['testing_score'].max()] 
bestlr_scratch.insert(0, 'Model', "lr_scratch")
bestlr_scratch #extracting the highest testing score


# In[31]:


#The following table does the comparison b/w the optimal testing  scores or R^2 for all the 5 models. 
#It also compares training score, intercept, coef_x_1 and coef_x_2 
total = []
total = pd.concat([linear, bestridge, bestlasso,bestelastic, bestlr_scratch ])
total.rename(index = {0: "formular"})

