# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 14:50:16 2021

@author: ST-01
"""

import numpy as np
import pandas as pd
import tensorflow as tf

df=pd.read_csv('D:/scale_data.csv')

x=df.drop(['F40'],axis=1)
y=df['F40']

from sklearn.preprocessing import StandardScaler
x=StandardScaler().fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=5)



#initializing the ANN
ann=tf.keras.models.Sequential()

#adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6,activation='tanh'))

#adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6,activation='tanh'))

#adding the output layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

#compiling the ANN
ann.compile(optimizer = 'adam',loss='binary_crossentropy',
            metrics=['accuracy'])

#Training the ANN
ann.fit(x_train,y_train,batch_size=20,epochs=50,verbose=1)

#Applying gridsearch
from sklearn.model_selection import GridSearchCV

param_grid=dict(batch_size= [10,20,30], epochs =[10,20,30,40,50], Optimizer_trial =['adam', 'rmsprop'])
                 
grid=GridSearchCV(estimator=ann,param_grid=param_grid,n_jobs=-1,cv=3,scoring='accuracy')
grid_result=grid.fit(x_train,y_train)