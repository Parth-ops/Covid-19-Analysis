#!/usr/bin/env python
# coding: utf-8

# # Importing pac

# In[81]:


### Data Collection
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr
import numpy as np
key="f7fee84c9ad6db319bb4e6cd55a797e5ab66561f"


# # Getting Tata Motors stock prices from Tiingo

# In[82]:


df = pdr.get_data_tiingo('TTM', api_key=key)


# In[83]:


df.to_csv('C:/Users/asus/TTM.csv')


# In[84]:


df=pd.read_csv('C:/Users/asus/TTM.csv')


# In[85]:


df.head()


# In[86]:


df.tail()


# In[87]:


df1=df.reset_index()['close']


# In[88]:


df1.shape


# # Plotting the current stock market graph

# In[89]:


plt.plot(df1)


# In[90]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[91]:


df1


# # Spliting the dataset into train and test 

# In[92]:


training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[93]:


training_size,test_size


# # Converting an array of values into a dataset matrix

# In[94]:


def create_dataset(dataset, time_step=1):
    dataX, dataY = [] , []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0] 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)


# In[95]:


#reshape into X=t,t+1,t+2,t+3 and Y=t+4
import numpy
time_step = 100
X_train , y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)


# In[96]:


print(X_train.shape), print(y_train.shape)


# In[97]:


print(X_test.shape), print(y_test.shape)


# # Reshaping input to be samples, time steps,features for LSTM

# In[98]:


X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# # Creating the STACKED LSTM model

# In[99]:



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[100]:


model = Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


# In[101]:


model.summary()


# # Training the model for 100 epochs

# In[102]:


model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64,verbose=1)


# In[116]:


import tensorflow as tf


# In[117]:


tf.__version__


# In[152]:


#check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[153]:


#Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# # Calculate RMSE performance metrics

# In[120]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[121]:


# Test Data RMSE
math.sqrt(mean_squared_error(y_test,test_predict))


# # Shift train predictions for plotting

# In[122]:


look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[123]:


len(test_data)


# In[137]:


x_input=test_data[341:].reshape(1,-1)
x_input.shape


# In[138]:


x_input=test_data[341:].reshape(1,-1)


# In[139]:


x_input.shape


# In[140]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# # Prediction for next 10 days

# In[141]:


from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[143]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[144]:


len(df1)


# In[145]:


df3 = df1.tolist()
df3.extend(lst_output)


# # Prediction for next 10 days on a graph

# In[150]:


plt.plot(day_new,scaler.inverse_transform(df1[1158:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# # Plotting the final predicted graph for Tata motors 

# In[151]:


df3 = df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])

