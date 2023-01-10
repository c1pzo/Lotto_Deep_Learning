#!/usr/bin/env python
# coding: utf-8

# # Cas de prédiction du Loto français 

# In[1]:


#Import des librairies utiles

import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, TimeDistributed, RepeatVector, Flatten
from keras.callbacks import EarlyStopping


# ## Fonction de scraping des tirages

# In[2]:


#fonction de scraping des tirages du loto
df_tirage = pd.read_csv('LOTTO_649.csv ', sep = ',', usecols=['num0','num1','num2','num3','num4','num5','num6','num7','num8','num9','num10','num11','num12','num13','num14','num15','num16','num17','num18','num19'])
dtype={'num':str }


# In[3]:


#A noter que plusieurs tirages se sont ajoutés dépuis le 21 : date de rédaction de l'article


# In[4]:


#sracping des tirages actuellement disponibles sur le site 

df_tirage[['num0','num1','num2','num3','num4','num5','num6','num7','num8','num9','num10','num11','num12','num13','num14','num15','num16','num17','num18','num19']].head()
#suppression  des tirages du super loto( A explorer later )
#df_tirage=df_tirage[(df_tirage['day']!='Vendredi') & (df_tirage['day']!='Mardi')]


# In[5]:


#df_tirage=df_tirage.tail(df_tirage.shape[0]-1))


# ## commentaires: 
# * le dernier tirage ici date du 07 décembre, ainsi afin de tester le modèle nous allons rétirer ce tirage du dataset dans la suite
# * Par contre on aurait évité de supprimer le tirage du 28 si on voulait prédire le prochain tirage ( celui du samedi 31)

# In[6]:


#df_tirage=df_tirage.tail(df_tirage.shape[0])# suppression du dernier tirage/à éviter selon le cas 
df_tirage.head()# le dernier tirage devient ici celui du 26


# ## Traitement  des données

# In[7]:


df = df_tirage.iloc[::-1]#inversion du dataframe pour placer le dernier tirage en dernière position
df = df[['num0','num1','num2','num3','num4','num5','num6','num7','num8','num9','num10','num11','num12','num13','num14','num15','num16','num17','num18','num19']]
#sélection des numéros à  traiter


# In[8]:


df.tail()# notre tirage du 26 ici devient le dernier de notre dataset afin de pourvoir organiser les data par historique


# In[9]:




#fonction de vérification de nombres en dessous d'une certaine valeur pour les 5 premiers numéros, sauf celui de chance
def is_under(data, number):
    return ((data['num0'] <= number).astype(int) + 
            (data['num1'] <= number).astype(int) +
            (data['num2'] <= number).astype(int) +
            (data['num3'] <= number).astype(int) +
            (data['num4'] <= number).astype(int) +
            (data['num5'] <= number).astype(int) + 
            (data['num6'] <= number).astype(int) +
            (data['num7'] <= number).astype(int) +
            (data['num8'] <= number).astype(int) +
            (data['num9'] <= number).astype(int) +
            (data['num10'] <= number).astype(int) + 
            (data['num11'] <= number).astype(int) +
            (data['num12'] <= number).astype(int) +
            (data['num13'] <= number).astype(int) +
            (data['num14'] <= number).astype(int) +
            (data['num15'] <= number).astype(int) + 
            (data['num16'] <= number).astype(int) +
            (data['num17'] <= number).astype(int) +
            (data['num18'] <= number).astype(int) +
            (data['num19'] <= number).astype(int))

#fonction de vérification de nombres pairs pour les 5 premiers numéros sauf celui de chance
def is_pair(data):
    return ((data['num0'].isin(pairs)).astype(int) + 
            (data['num1'].isin(pairs)).astype(int) +
            (data['num2'].isin(pairs)).astype(int) +
            (data['num3'].isin(pairs)).astype(int) +
            (data['num4'].isin(pairs)).astype(int) +
            (data['num5'].isin(pairs)).astype(int) + 
            (data['num6'].isin(pairs)).astype(int) +
            (data['num7'].isin(pairs)).astype(int) +
            (data['num8'].isin(pairs)).astype(int) +
            (data['num9'].isin(pairs)).astype(int) +
            (data['num10'].isin(pairs)).astype(int) + 
            (data['num11'].isin(pairs)).astype(int) +
            (data['num12'].isin(pairs)).astype(int) +
            (data['num13'].isin(pairs)).astype(int) +
            (data['num14'].isin(pairs)).astype(int) +
            (data['num15'].isin(pairs)).astype(int) + 
            (data['num16'].isin(pairs)).astype(int) +
            (data['num17'].isin(pairs)).astype(int) +
            (data['num18'].isin(pairs)).astype(int) +
            (data['num19'].isin(pairs)).astype(int))

#fonction de vérification de nombres impairs pour les 5 premiers numéros sauf celui de chance
def is_impair(data):
    return ((data['num0'].isin(impairs)).astype(int) + 
            (data['num1'].isin(impairs)).astype(int) +
            (data['num2'].isin(impairs)).astype(int) +
            (data['num3'].isin(impairs)).astype(int) +
            (data['num4'].isin(impairs)).astype(int) +
            (data['num5'].isin(impairs)).astype(int) + 
            (data['num6'].isin(impairs)).astype(int) +
            (data['num7'].isin(impairs)).astype(int) +
            (data['num8'].isin(impairs)).astype(int) +
            (data['num9'].isin(impairs)).astype(int) +
            (data['num10'].isin(impairs)).astype(int) + 
            (data['num11'].isin(impairs)).astype(int) +
            (data['num12'].isin(impairs)).astype(int) +
            (data['num13'].isin(impairs)).astype(int) +
            (data['num14'].isin(impairs)).astype(int) +
            (data['num15'].isin(impairs)).astype(int) + 
            (data['num16'].isin(impairs)).astype(int) +
            (data['num17'].isin(impairs)).astype(int) +
            (data['num18'].isin(impairs)).astype(int) +
            (data['num19'].isin(impairs)).astype(int))



#liste de nombres pairs et impairs
pairs = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70]
impairs = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69]

#Fonction de calcul de la somme de la différence au carré des 5 premiers numéros, sauf celui de chance
def sum_diff(data):
    return ((data['num1'] - data['num0'])**2 + 
            (data['num2'] - data['num1'])**2 +
            (data['num3'] - data['num2'])**2 +
            (data['num4'] - data['num3'])**2 +
            (data['num5'] - data['num4'])**2 + 
            (data['num6'] - data['num5'])**2 +
            (data['num7'] - data['num6'])**2 +
            (data['num8'] - data['num7'])**2 +
            (data['num9'] - data['num8'])**2 + 
            (data['num10'] - data['num9'])**2 +
            (data['num11'] - data['num10'])**2 +
            (data['num12'] - data['num11'])**2 +
            (data['num13'] - data['num12'])**2 + 
            (data['num14'] - data['num13'])**2 +
            (data['num15'] - data['num14'])**2 +
            (data['num16'] - data['num15'])**2 +
            (data['num17'] - data['num16'])**2 + 
            (data['num18'] - data['num17'])**2 +
            (data['num19'] - data['num18'])**2)

# Calcul de la fréquence de tirage de chaque numéro
freqs = []
for val in range(70):
    count = ( (df['num0'] == val+1).sum() +
              (df['num1'] == val+1).sum() +
              (df['num2'] == val+1).sum() +
              (df['num3'] == val+1).sum() +
              (df['num4'] == val+1).sum() +
              (df['num5'] == val+1).sum() +
              (df['num6'] == val+1).sum() +
              (df['num7'] == val+1).sum() +
              (df['num8'] == val+1).sum() +
              (df['num9'] == val+1).sum() +
              (df['num10'] == val+1).sum() +
              (df['num11'] == val+1).sum() +
              (df['num12'] == val+1).sum() +
              (df['num13'] == val+1).sum() +
              (df['num14'] == val+1).sum() +
              (df['num15'] == val+1).sum() +
              (df['num16'] == val+1).sum() +
              (df['num17'] == val+1).sum() +
              (df['num18'] == val+1).sum() +
              (df['num19'] == val+1).sum() )
    
    freqs.append(count)
ax = plt.gca() ;  ax.invert_yaxis()
plt.gcf().set_size_inches(7, 6)
heatmap = plt.pcolor(np.reshape(np.array(freqs), (7, 10)), cmap=plt.cm.Blues)

def freq_val(data, column):
    tab = data[column].values.tolist()
    freqs = []
    pos = 1
    for e in tab:
        freqs.append(tab[0:pos].count(e))
        pos = pos + 1
    return freqs



df['sum'] = ((df.num0 + df.num1 + df.num2 + df.num3 + df.num4 + df.num5 + df.num6 + df.num7 + df.num8 + df.num9 +df.num10 + df.num11 + df.num12 + df.num13 + df.num14 +df.num15 + df.num16 + df.num17 + df.num18 + df.num19 ) >1400).astype(int)


# In[10]:


#ajout de la difference entre les numéros(A explorer ASAp)
for i in range(4):
    (i,i+1)
df['diff_{}'.format(i)]=df['num{}'.format(i+1)]-df['num{}'.format(i)]
#application des fonctions sur le dataframe
df['freq_num0'] = freq_val(df, 'num0')
df['freq_num1'] = freq_val(df, 'num1')
df['freq_num2'] = freq_val(df, 'num2')
df['freq_num3'] = freq_val(df, 'num3')
df['freq_num4'] = freq_val(df, 'num4')
df['freq_num5'] = freq_val(df, 'num5')
df['freq_num6'] = freq_val(df, 'num6')
df['freq_num7'] = freq_val(df, 'num7')
df['freq_num8'] = freq_val(df, 'num8')
df['freq_num9'] = freq_val(df, 'num9')
df['freq_num10'] = freq_val(df, 'num10')
df['freq_num11'] = freq_val(df, 'num11')
df['freq_num12'] = freq_val(df, 'num12')
df['freq_num13'] = freq_val(df, 'num13')
df['freq_num14'] = freq_val(df, 'num14')
df['freq_num15'] = freq_val(df, 'num15')
df['freq_num16'] = freq_val(df, 'num16')
df['freq_num17'] = freq_val(df, 'num17')
df['freq_num18'] = freq_val(df, 'num18')
df['freq_num19'] = freq_val(df, 'num19')
df['sum_diff'] = sum_diff(df)#somme de la différence au carré entre chaque couple de numéros successifs dans le tirage
df['pair'] = is_pair(df)
df['impair'] = is_impair(df)#verification de nombre pair et impair
df['is_under_24'] = is_under(df, 24)  # Les numeros en dessous de 24 
df['is_under_48'] = is_under(df, 48)# Les numeros en dessous de 40 
df.head(6)


# ## Modèle et fonction de formatage des données en entrée du LSTM

# In[11]:


#capture 3: fonction define model seulement


# In[12]:


# j'ai ici défini plusieurs modèles à tester mais pour l'intant je tavaille avec le lstm(fonction : define_model)
# j'ai ici défini window_length à 12 pour apprendre sur 1 mois de données 

#Params du modèle
nb_label_feature=20

UNITS = 250
BATCHSIZE = 45
EPOCH = 1500
#ACTIVATION = "softmax"
OPTIMIZER ='adam' # rmsprop, adam, sgd
LOSS = 'mae'#'categorical_crossentropy' #mse
DROPOUT = 0.11
window_length =23 #12 
number_of_features = df.shape[1]

#Architecture du modèle
def define_model(number_of_features,nb_label_feature):
    #initialisation du rnn
    model = Sequential()
    #ajout de la premiere couche lstm
    model.add(LSTM(UNITS, input_shape=(window_length, number_of_features), return_sequences=True))
    model.add(LSTM(UNITS, dropout=0.1, return_sequences=False))
    #ajout de la couche de sortie
    model.add(Dense(nb_label_feature))
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=['acc'])
    return model

def define_bidirectionnel_model(number_of_features,nb_label_feature):
    model = Sequential()
    model.add(Bidirectional(LSTM(100, dropout=0.2, return_sequences=True), input_shape=(window_length, number_of_features)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(100, dropout=0.1))
    model.add(Dense(nb_label_feature))
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=['acc'])
    return model

def define_autoencoder_model(number_of_features,nb_label_feature):
    model = Sequential()
    model.add(LSTM(100, input_shape=(window_length, number_of_features), return_sequences=True))
    model.add(LSTM(50, return_sequences=False))
    model.add(RepeatVector(window_length))
    model.add(LSTM(100, dropout=0.1, return_sequences=True))
    model.add(LSTM(50, return_sequences=True))
    model.add(TimeDistributed(Dense(number_of_features)))
    model.add(Flatten())
    model.add(Dense(nb_label_feature))
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=['acc'])
    return model



#model = define_model(number_of_features,nb_label_feature)
#model3 = define_autoencoder_model(number_of_features,nb_label_feature)
#model4 = define_bidirectionnel_model(number_of_features,nb_label_feature)

#Moniteur pour stoper le training
es = EarlyStopping(monitor='acc', mode='max', verbose=1, patience=100)


# In[13]:


# Fonction de formatage des données en entrée du LSTM
def create_lstm_dataset(df, window_length,nb_label_feature):
    number_of_rows = df.shape[0]   #taille du dataset number_of_features
    number_of_features = df.shape[1]
    scaler = StandardScaler().fit(df.values)
    transformed_dataset = scaler.transform(df.values)
    transformed_df = pd.DataFrame(data=transformed_dataset, index=df.index)
    #tableau de tableau de taille(number_of_rows-window_length) et window_length ligne,number_of_features
    #lstm:[nb total de row ,nb de ligne dans le passé, nb de colonne(feature)]
    train = np.empty([number_of_rows-window_length, window_length, number_of_features], dtype=float)
    
    label = np.empty([number_of_rows-window_length, nb_label_feature], dtype=float)
    for i in range(0, number_of_rows-window_length):
        train[i] = transformed_df.iloc[i:i+window_length, 0: number_of_features]
        label[i] = transformed_df.iloc[i+window_length: i+window_length+1, 0:nb_label_feature]
        
    #définition du modèle Lstm  
    model = define_model(number_of_features,nb_label_feature)
        
    return train, label, model,scaler


# ## Training

# In[14]:


#formatage des données
train, label,model,scaler1 = create_lstm_dataset(df, window_length,nb_label_feature)
print(train.shape)
print(label.shape)
 


# * On voit ici que notre dataset d'entrainement après formatage est constitué de 1911 vecteurs contenant chacun 12 tirages où chaque tirage contient 19 features calculés plus haut
# 
# * Quant aux labels, on a bien 1911 vecteurs de 6 features soit les 6 numéros de chaque tirages
# 
# * Ainsi à partir des 12 tirages précédent on éssaie de prédire le tirage suivant lors de l'entrainement

# In[15]:


#Training
history=model.fit(train, label, batch_size=BATCHSIZE, epochs=EPOCH, verbose=2, callbacks=[es])


# ## Fonction de perte 

# In[16]:


#capture 6


# In[17]:


plt.plot(history.history['loss'])
plt.legend(['train_loss'])
plt.show()


# ## Prédiction du tirage suivant le dernier tirage de notre dataset de train

# In[18]:


#Prediction basée sur les 12 derniers tirages
last_twelve = df.tail(window_length) # on recupere les 12 derniers tirages
scaler = StandardScaler().fit(df.values)
scaled_to_predict = scaler.transform(last_twelve)
scaled_predicted_output_1 = model.predict(np.array([scaled_to_predict]))


# In[19]:


#prediction
tom = df.tail(window_length).iloc[:,0:20] # 
scaler = StandardScaler().fit(df.iloc[:,0:20])
scaled_to_predict = scaler.transform(tom)
print(scaler.inverse_transform(scaled_predicted_output_1).astype(int)[0])


# In[ ]:




