# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 16:32:53 2021

@author: ST-01
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import metrics
from sklearn.linear_model import  LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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

df=pd.read_csv('D:/scale_data.csv')
x=df.drop(['F40'],axis=1)
y=df['F40']


from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=5)

import xgboost
from xgboost import XGBClassifier
classifier=XGBClassifier()   
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
y_pred_train=classifier.predict(x_train)
print('Train accuracy',accuracy_score(y_train,y_pred_train))
print('Test accuracy',accuracy_score(y_test,y_pred))