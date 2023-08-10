#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
        residualError =   X.dot(theta)-y
        gradient =  transposedX.dot(residualError)/m
        change = [alpha * x for x in gradient]
        theta = np.subtract(theta, change)  # or theta = theta - alpha * gradient

        ################PLACEHOLDER4 #end##########################

        ################PLACEHOLDER5 #start##########################
        # calculate the current cost with the new theta; 
        atmp  = (1 / 2*m) * np.sum(residualError ** 2)  
        print(atmp)
        arrCost.append(atmp)
        # cost = (1 / m) * np.sum(residualError ** 2)
     
        ################PLACEHOLDER5 #start##########################

    return theta, arrCost

