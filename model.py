# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 16:26:07 2021

@author: ST-01
"""

import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

df=pd.read_csv('D:/scale_data.csv')
x=df.drop(['F40'],axis=1)
y=df['F40']

accuracy={}
speed={}

#LogisticRegression

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

start=time()
cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=2,random_state=0)
score=cross_val_score(model,x,y,scoring="accuracy",cv=cv,n_jobs=-1)

speed["LogisticRegression"]=np.round(time()-start,3)
accuracy["LogisticRegression"]=np.mean(score).round(3)

print(f"Mean Accuracy: {accuracy['LogisticRegression']}\nStd:{np.std(score):.3f}\nRun time:{speed['LogisticRegression']}s"
     )

#Support Vector Machine

from sklearn.svm import SVC

model=SVC()

start=time()
cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=2,random_state=0)
score=cross_val_score(model,x,y,scoring="accuracy",cv=cv,n_jobs=-1)

speed["SVC"]=np.round(time()-start,3)
accuracy["SVC"]=np.mean(score).round(3)

print(f"Mean Accuracy: {accuracy['SVC']}\nStd:{np.std(score):.3f}\nRun time:{speed['SVC']}s"
     )

#Artifical Neural Network

from sklearn.neural_network import MLPClassifier

model = MLPClassifier()

start=time()
cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=2,random_state=0)
score=cross_val_score(model,x,y,scoring="accuracy",cv=cv,n_jobs=-1)

speed["MLP"]=np.round(time()-start,3)
accuracy["MLP"]=np.mean(score).round(3)

print(f"Mean Accuracy: {accuracy['MLP']}\nStd:{np.std(score):.3f}\nRun time:{speed['MLP']}s"
     )

#Random forest

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

start=time()
cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=2,random_state=0)
score=cross_val_score(model,x,y,scoring="accuracy",cv=cv,n_jobs=-1)

speed["RF"]=np.round(time()-start,3)
accuracy["RF"]=np.mean(score).round(3)

print(f"Mean Accuracy: {accuracy['RF']}\nStd:{np.std(score):.3f}\nRun time:{speed['RF']}s"
     )

#Gradient Boosting.....................

from sklearn.ensemble import GradientBoostingClassifier

model=GradientBoostingClassifier()

start=time()
cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=2,random_state=0)
score=cross_val_score(model,x,y,scoring="accuracy",cv=cv,n_jobs=-1)

speed["GradientBoosting"]=np.round(time()-start,3)
accuracy["GradientBoosting"]=np.mean(score).round(3)

print(f"Mean Accuracy: {accuracy['GradientBoosting']}\nStd:{np.std(score):.3f}\nRun time:{speed['GradientBoosting']}s"
     )

#XGBOOST................

from xgboost import XGBClassifier

model=XGBClassifier()

start=time()
cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=2,random_state=0)
score=cross_val_score(model,x,y,scoring="accuracy",cv=cv,n_jobs=-1)

speed["XGB"]=np.round(time()-start,3)
accuracy["XGB"]=np.mean(score).round(3)

print(f"Mean Accuracy: {accuracy['XGB']}\nStd:{np.std(score):.3f}\nRun time:{speed['XGB']}s"
     )

#CATBOOST.............

from catboost import CatBoostClassifier 

model=CatBoostClassifier()

start=time()
cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=2,random_state=0)
score=cross_val_score(model,x,y,scoring="accuracy",cv=cv,n_jobs=-1)

speed["CatBoost"]=np.round(time()-start,3)
accuracy["CatBoost"]=np.mean(score).round(3)

print(f"Mean Accuracy: {accuracy['CatBoost']}\nStd:{np.std(score):.3f}\nRun time:{speed['CatBoost']}s"
     )

