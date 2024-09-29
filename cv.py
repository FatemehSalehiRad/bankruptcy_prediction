# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 16:03:51 2021

@author: ST-01
"""

import pandas as pd
import xgboost
from xgboost import XGBClassifier
from sklearn import metrics#for checking the model accuracy

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

from sklearn.model_selection import KFold
classifier=XGBClassifier(max_depth=2) 
kfold_validation=KFold(10)

import numpy as np
from sklearn.model_selection import cross_val_score
results=cross_val_score(classifier,x,y,cv=kfold_validation,scoring='recall')
print(results)
print(np.mean(results))

