# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 14:13:48 2021

@author: ST-01
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import sklearn

df=pd.read_csv('D:/scale_data.csv')

from sklearn.linear_model import LogisticRegression#for Logisic Regression Algorithm
from sklearn.model_selection import train_test_split#to split the dataset for training and testing
from sklearn import metrics#for checking the model accuracy


x=df.drop(['F40'],axis=1)
y=df['F40']

x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=5)

logreg=LogisticRegression()
logreg.fit(x_train,y_train)

y_pred=logreg.predict(x_test)
#confusion_matrix
from sklearn.metrics import confusion_matrix
#accuracy
from sklearn.metrics import accuracy_score
#recall_score
from sklearn.metrics import recall_score
#precision_score
from sklearn.metrics import precision_score
#F1_score
from sklearn.metrics import f1_score

a1=confusion_matrix(y_test,y_pred)
a2=accuracy_score(y_test,y_pred)
a3=recall_score(y_test,y_pred)
a4=precision_score(y_test,y_pred)
a5=f1_score(y_test,y_pred)