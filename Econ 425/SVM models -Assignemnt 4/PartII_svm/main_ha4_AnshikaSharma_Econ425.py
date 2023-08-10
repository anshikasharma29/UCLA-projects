#!/usr/bin/env python
# coding: utf-8

# ## <center> Homework4- Part-1
# ##   <center> Anshika Sharma, UCLA ID:305488635

# In[1]:


#importing all the packages
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import accuracy_score, mean_squared_error 

print("version: ", tf.__version__)
print("Hub bersion: ", hub.__version__)


# In[2]:


#load dataset
from sklearn.datasets import fetch_california_housing

ca_house_db = fetch_california_housing()
print(ca_house_db.data.shape)
print(ca_house_db.target.shape)
print(ca_house_db.feature_names)
print(ca_house_db.DESCR)
print(ca_house_db)


# In[3]:


#setting the seed (optional)
from numpy.random import seed
seed(0)
tf.random.set_seed(0)


# In[4]:


#######################Step 1 Data splitting################################
Y = ca_house_db.target
X = ca_house_db.data

#split samples into two halves (traning process and testing process)
X_a, X_test, Y_a, Y_test = train_test_split(X,Y,train_size = 0.5,random_state = 0)

#subset of 2064 samples for validation purpose and the rest 8256 samples for training
X_train, X_valid, Y_train, Y_valid = train_test_split(X_a,Y_a,train_size = (8256/X_a.shape[0]),random_state = 0)
#############################end Step 1#####################################


# In[5]:


#######################Step 2 Model selection################################
#Specify at least three sets of hyper-parameters 
#Parameter includes :
#1.number of hidden layers 
#2.learning rate 
#3.activation function

# define the keras model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
seed_value = 0
tf.random.set_seed(seed_value)

mod1 = Sequential()
mod1.add(Dense(15,activation='relu')) #hidden layer1
mod1.add(Dense(15,activation='relu')) #hidden layer2
mod1.add(Dense(1)) #outputlayer
mod1.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss='mse') #learning_rate=0.01


# In[6]:


tf.random.set_seed(seed_value)
mod2 = Sequential()
mod2.add(Dense(15,activation='relu')) 
mod2.add(Dense(15,activation='sigmoid')) 
mod2.add(Dense(1)) 
mod2.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss='mse') 


# In[7]:


tf.random.set_seed(seed_value)
mod3 = Sequential()
mod3.add(Dense(15,activation='tanh')) 
mod3.add(Dense(15,activation='softmax')) 
mod3.add(Dense(1)) 
mod3.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss='mse')


# In[8]:


tf.random.set_seed(seed_value)
mod4 = Sequential()
mod4.add(Dense(15,activation='relu')) 
mod4.add(Dense(15,activation='relu')) 
mod4.add(Dense(1))
mod4.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),loss='mse') #learning_rate=0.0001


# In[9]:


tf.random.set_seed(seed_value)
mod5 = Sequential()
mod5.add(Dense(15,activation='relu')) #hidden layer1
mod5.add(Dense(15,activation='relu')) #hidden layer1
mod5.add(Dense(15,activation='relu')) #hidden layer1
mod5.add(Dense(1)) #outputlayer
mod5.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss='mse') #learning_rate=0.001


# In[10]:


tf.random.set_seed(seed_value)
mod6 = Sequential()
mod6.add(Dense(20,activation='relu')) #hidden layer1, activation fn = "relu"
mod6.add(Dense(20,activation='relu')) #hidden layer1, activation fn = "relu"
mod6.add(Dense(20,activation='sigmoid')) #hidden layer1, activation fn = "sigmoid"
mod6.add(Dense(1))  #outputlayer
mod6.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss='mse') #learning_rate=0.001


# In[11]:


# Train the the FNN model on the training samples
#iteration. = samplesize/batch size
batch_size = 32
epochs = 200
tf.random.set_seed(seed_value)

history1 = mod1.fit(x=X_train,y=Y_train,
          validation_data=(X_valid,Y_valid),
          batch_size=128,epochs=400)


# In[12]:


tf.random.set_seed(seed_value)
history2 = mod2.fit(x=X_train,y=Y_train,
          validation_data=(X_valid,Y_valid),
          batch_size=128,epochs=400)


# In[13]:


tf.random.set_seed(seed_value)
history3 = mod3.fit(x=X_train,y=Y_train,
          validation_data=(X_valid,Y_valid),
          batch_size=128,epochs=400)


# In[14]:


tf.random.set_seed(seed_value)
history4 = mod4.fit(x=X_train,y=Y_train,
          validation_data=(X_valid,Y_valid),
          batch_size=128,epochs=400)


# In[15]:


tf.random.set_seed(seed_value)
history5 = mod5.fit(x=X_train,y=Y_train,
          validation_data=(X_valid,Y_valid),
          batch_size=128,epochs=400)


# In[16]:


tf.random.set_seed(seed_value)
history6 = mod6.fit(x=X_train,y=Y_train,
          validation_data=(X_valid,Y_valid),
          batch_size=128,epochs=400)


# In[17]:


mod1.summary()
mod2.summary()
mod3.summary()
mod4.summary()
mod5.summary()
mod6.summary()


# In[45]:


#R2_score of the six models with different parameter
import pandas as pd
from sklearn.metrics import r2_score

col = ['R2_score'] 
ind = ['mod_%.0g'%i for i in range(1,7)]
summary1 = pd.DataFrame(index=ind, columns=col) #summary table

j = 0
for i in [mod1,mod2,mod3,mod4,mod5,mod6]: 
    y_predv = i.predict(X_valid)
    summary1.iloc[j,0] = r2_score(Y_valid, y_predv) #R2_score when applying on validation set
    j = j+1
summary1

#Details of each models
des = {'Hidden_layer': [2, 2,2,2,3,3], 'activation_function': ['relu,relu', 'relu, sigmoid','tanh, softmax','relu,relu','relu,relu,relu', 'relu,relu,sigmoid'] , 'Learning_rate': [0.01, 0.01, 0.01, 0.0001,0.001,0.001]}
desdf = pd.DataFrame(des, index=ind)

#Combine two tables (performance and Details of each models)
summaryfinal = pd.concat([summary1 ,desdf], axis = 1 )

summaryfinal

#############################end Step 2#####################################


# In[37]:


#######################Step 3 Apply the model on testing set#############################
col = ['R2_score']
ind = ['mod_%.0g'%i for i in [6]]
summary2 = pd.DataFrame(index=ind, columns=col)

j = 0
for i in [mod6]:
    y_pred = i.predict(X_test) #applying models on testing set
    summary2.iloc[j,0] = r2_score(Y_test, y_pred) #R2_score when applying on testing set
    j = j+1
summary2
#############################end Step 3#####################################


# In[38]:


#######################Step 4 Analyze the testing results #############################
Y_pred6 = mod6.predict(X_test)
error = []
for i in range(X_test.shape[0]):
    errors = abs(Y_pred6[i,0]- Y_test[i])  #Calucate absolute value of error
    error.append(errors )


# In[39]:


summary3 = pd.DataFrame(
{'Y': Y_test,
 'Y_hat': Y_pred6[:,0], 
 'abs(error)': error  # Calucate absolute value of error
}
)


# In[40]:


X_testdf = pd.DataFrame(X_test, columns= ca_house_db.feature_names )
total = pd.concat([summary3 ,X_testdf], axis = 1 ) #merge data set
final_table  = total.sort_values(by='abs(error)', ascending=False).head(10) #sorting the dataframe following to abs of error


# In[41]:


#Average/Meadian/max/min of each features.
avg = pd.DataFrame(total.mean(axis = 0)).T #mean
med = pd.DataFrame(total.median(axis = 0)).T 
maxs = pd.DataFrame(total.max(axis = 0)).T 
mins = pd.DataFrame(total.min(axis = 0)).T
final_table.loc["mean_sample",:] = avg.iloc[0,:] 
final_table.loc["median_sample",:] = med.iloc[0,:] 
final_table.loc["max_sample",:] = maxs.iloc[0,:]
final_table.loc["min_sample",:] = mins.iloc[0,:]


# In[42]:


final_table


# In[25]:


############end Step 4############

