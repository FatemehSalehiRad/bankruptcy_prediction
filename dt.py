# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 16:35:27 2021

@author: ST-01
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import metrics

df=pd.read_csv('D:/scale_data.csv')

x=df.drop(['F40'],axis=1)
y=df['F40']

from sklearn.linear_model import  LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DT=DecisionTreeClassifier(criterion='gini',max_depth=3,random_state=3)

from sklearn.preprocessing import StandardScaler
x=StandardScaler().fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=5)

DT.fit(x_train,y_train)

y_pred=DT.predict(x_test)

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

print('The accuracy of DT is:',(metrics.accuracy_score(y_test,y_pred)))