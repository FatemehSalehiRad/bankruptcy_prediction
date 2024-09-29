# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 16:08:59 2021

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
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

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

params={
    "learning_rate"   :[0.05,0.1,0.15,0.2,0.25,0.3],
    "max_depth"       :[3,4,5,6,8,10,12,15],
    "min_child_weight":[1,3,5,7],
    "gamma"           :[0,0.1,0.2,0.3,0.4],
    "colsample_bytree":[0.3,0.4,0.5,0.7],
    "n_estimators"    :[1,2,3,4,5],
    "sub_sample"      :[0.1,0.2,0.3,1],
}



classifier=XGBClassifier()
random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5)
random_search.fit(x_train,y_train)

#classifier=XGBClassifier()
#grid_search=GridSearchCV(classifier,param_grid=params,scoring='roc_auc',n_jobs=-1,cv=5)
#grid_search.fit(x_train,y_train)

best_random_grid=random_search.best_estimator_
#best_serrch_grid=grid_search.best_estimator_


y_pred=best_random_grid.predict(x_test)
#y_pred=best_serrch_grid.predict(x_test)