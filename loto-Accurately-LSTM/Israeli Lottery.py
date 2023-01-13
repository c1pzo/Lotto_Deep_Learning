#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# In[5]:


df = pd.read_csv("LOTTO_MAX.csv")


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df.drop(['Game', 'Date'], axis=1, inplace=True)


# In[11]:


df.head()


# In[12]:


scaler = StandardScaler().fit(df.values)
transformed_dataset = scaler.transform(df.values)
transformed_df = pd.DataFrame(data=transformed_dataset, index=df.index)


# In[13]:


transformed_df.head()


# In[14]:


# All our games
number_of_rows = df.values.shape[0]
number_of_rows


# In[15]:


# Amount of games we need to take into consideration for prediction
window_length = 7
window_length 


# In[16]:


# Balls counts
number_of_features = df.values.shape[1]
number_of_features


# In[17]:


X = np.empty([ number_of_rows - window_length, window_length, number_of_features], dtype=float)
X


# In[18]:


y = np.empty([ number_of_rows - window_length, number_of_features], dtype=float)
y


# In[19]:


for i in range(0, number_of_rows-window_length):
    X[i] = transformed_df.iloc[i : i+window_length, 0 : number_of_features]
    y[i] = transformed_df.iloc[i+window_length : i+window_length+1, 0 : number_of_features]


# In[20]:


X.shape


# In[21]:


y.shape


# In[22]:


X[0]


# In[23]:


X[1]


# In[24]:


y[0]


# In[25]:


y[1]


# In[26]:


# Recurrent Neural Netowrk (RNN) with Long Short Term Memory (LSTM)
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout
batch_size = 100


# In[27]:


# Initialising the RNN
model = Sequential()
# Adding the input layer and the LSTM layer
model.add(Bidirectional(LSTM(240,
                        input_shape = (window_length, number_of_features),
                        return_sequences = True)))
# Adding a first Dropout layer
model.add(Dropout(0.2))
# Adding a second LSTM layer
model.add(Bidirectional(LSTM(240,
                        input_shape = (window_length, number_of_features),
                        return_sequences = True)))
# Adding a second Dropout layer
model.add(Dropout(0.2))
# Adding a third LSTM layer
model.add(Bidirectional(LSTM(240,
                        input_shape = (window_length, number_of_features),
                        return_sequences = True)))
# Adding a fourth LSTM layer
model.add(Bidirectional(LSTM(240,
                        input_shape = (window_length, number_of_features),
                        return_sequences = False)))
# Adding a fourth Dropout layer
model.add(Dropout(0.2))
# Adding the first output layer
model.add(Dense(59))
# Adding the last output layer
model.add(Dense(number_of_features))


# In[36]:


from tensorflow import keras
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.0001), loss ='mse', metrics=['accuracy'])


# In[38]:


model.fit(x=X, y=y, batch_size=100, epochs=300, verbose=2)


# In[39]:


to_predict = df.tail(8)
to_predict.drop([to_predict.index[-1]],axis=0, inplace=True)
to_predict


# In[40]:


to_predict = np.array(to_predict)
to_predict


# In[41]:


scaled_to_predict = scaler.transform(to_predict)


# In[42]:


y_pred = model.predict(np.array([scaled_to_predict]))
print("The predicted numbers in the last lottery game are:", scaler.inverse_transform(y_pred).astype(int)[0])


# In[43]:


prediction = df.tail(1)
prediction = np.array(prediction)
print("The actual numbers in the last lottery game were:", prediction[0])

