#!/usr/bin/env python
# coding: utf-8

# ### <center> Econ 425 Homework Assignment 1, Part-1 
# ### <center> Anshika Sharma, UCLA ID(305488635)

# In[1]:


from download_data import download_data #from the download_data.py file, I imported the function "download_data"
import numpy as np
import matplotlib.pyplot as plt
#from GD import gradientDescent ## note that instead of importing GD, I copy-pasted and used the GD function in this file
from dataNormalization import meanNormalization #from the datanormalization.py file, I imported these function.
from dataNormalization import rescaleMatrix     
from dataNormalization import rescaleNormalization


# In[2]:


#NOTICE: Fill in the codes between "%PLACEHOLDER#start" and "PLACEHOLDER#end"

# There are two PLACEHODERS IN THIS SCRIPT

# parameters

################PLACEHOLDER1 #start##########################
# test multiple learning rates and report their convergence curves. 
ALPHA = 0.1
MAX_ITER = 500
#Note: I am specifying the various alpha and max_iter that I took and adding it as a comment:
#ALPHA = 0.1
#MAX_ITER = 500

#ALPHA = 0.7
#MAX_ITER = 500

#ALPHA = 0.03
#MAX_ITER = 500

#ALPHA = 0.03
#MAX_ITER = 45

#ALPHA = 0.03
#MAX_ITER = 750
################PLACEHOLDER1 #end##########################


# In[3]:


#% step-1: load data and divide it into two subsets, used for training and testing
sat = download_data('sat.csv', [1, 2, 4]).values # three columns: MATH SAT, VERB SAT, UNI. GPA  # convert frame to matrix


# In[4]:


################PLACEHOLDER2 #start##########################
# Normalize data
sat1 = sat.copy()
for i in range(3):
    sat1[:,i] = meanNormalization(sat[:,i])
sat1


sat2 = sat.copy()
for i in range(3):
    sat2[:,i] = rescaleNormalization(sat[:,i])
sat2


sat3 = rescaleMatrix(sat)
#Note that since I used the all the three normalization techniques, I used sat1,2,3 accordingly turn by turn to get the results
################PLACEHOLDER2 #end##########################


# In[5]:


#splitting into training and testing data
# training data;
satTrain1 = sat1[0:60, :]
satTrain2 = sat2[0:60, :]
satTrain3 = sat3[0:60, :]
# testing data; 
satTest1 = sat1[60:len(sat),:]
satTest2 = sat2[60:len(sat),:]
satTest3 = sat3[60:len(sat),:]


# In[6]:


#% step-2: train a linear regression model using the Gradient Descent (GD) method
# ** theta and xValues have 3 columns since have 2 features: y = (theta * x^0) + (theta * x^1) + (theta * x^2)
theta = np.zeros(3) 
#print(theta.shape)
xValues = np.ones((60, 3)) 
xValues[:, 1:3] = satTrain1[:, 0:2] #Note that "satTrain1" can be changed to "Sat Train2" & "Sat Train3" to see the results of different Normalization techniques
yValues = satTrain1[:, 2] # same thing as above!

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
    m = len(y) #no. of training samples
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

[theta, arrCost] = gradientDescent(xValues, yValues, theta, ALPHA, MAX_ITER)

 
#visualize the convergence curve
plt.plot(range(0,len(arrCost)),arrCost);
plt.xlabel('iteration')
plt.ylabel('cost')
plt.title('alpha = {}  theta = {}'.format(ALPHA, theta))
plt.show()


# In[7]:


#% step-3: testing
testXValues = np.ones((len(satTest1), 3)) #Note that "satTest1" can be changed to "SatTest2" & "SatTest3" to see the results of different Normalization techniques
testXValues[:, 1:3] = satTest1[:, 0:2] 
tVal =  testXValues.dot(theta) #predicted gpa


# In[8]:


#% step-4: evaluation
# calculate average error and standard deviation
tError = np.sqrt([x**2 for x in np.subtract(tVal, satTest1[:, 2])])
print('results: {} ({})'.format(np.mean(tError), np.std(tError)))
################PLACEHOLDER3 #start##########################
# Calculate and print R2 
s1 = np.subtract(satTest1[:, 2],tVal) #Note that "satTest1" can be changed to "SatTest2" & "SatTest3" to see the results of different Normalization techniques
s11 = 0
for i in s1:
    s11 = s11 + (i*i)

s2 = satTest1[:, 2].mean() - satTest1[:, 2]
s22 = 0
for i in s2:
    s22 = s22 + (i*i)
    
(s22-s11)/s22
print("R2 is {}".format((s22-s11)/s22))

# Note the following code can also be used to calculate R2 and gives the exact same result but I have put it in comments form. 

# from sklearn.metrics import r2_score
# R2test_r2=  r2_score(satTest1[:, 2],tVal)
# print("R2 is {}".format(R2test_r2))
################PLACEHOLDER3 #end##########################

