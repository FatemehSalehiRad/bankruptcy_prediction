# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 16:31:50 2021

@author: ST-01
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

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

#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC
#from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import GradientBoostingClassifier
#from xgboost import XGBClassifier
#from catboost import CatBoostClassifier 

#model=LogisticRegression()
#model=SVC()
#model = MLPClassifier()
model = RandomForestClassifier()
#model=GradientBoostingClassifier()
#model=XGBClassifier()
#model=CatBoostClassifier()

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=15)

model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred_train=model.predict(x_train)
print('Train accuracy',accuracy_score(y_train,y_pred_train))
print('Test accuracy',accuracy_score(y_test,y_pred))

# Randomized Search CV

from sklearn.model_selection import RandomizedSearchCV
#Number of trees in a RandomForest
n_estimators=[int(x) for  x  in np.linspace(start=200,stop=2000,num=10)]
#Number of features to consider at every split
max_features=['auto','sqrt','log2']
#Max Number of levels in a tree
max_depth=[int(x) for x in np.linspace(10,1000,10)]
#Min Number of samples required to split a node
min_samples_split=[2,5,10,14]
##Min Number of samples required at each leaf node
min_samples_leaf=[1,2,4,5,7,9]
#create the random grid
random_grid={'n_estimators'     :n_estimators,
             'max_features'     :max_features,
             'max_depth'        :max_depth,
             'min_samples_split':min_samples_split,
             'min_samples_leaf' :min_samples_leaf,
             'criterion'      :['entropy','gini']
}

random_search=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=100,scoring='roc_auc',n_jobs=-1,cv=3)
random_search.fit(x_train,y_train)

best_random_grid=random_search.best_estimator_
#best_serrch_grid=grid_search.best_estimator_

y_pred=best_random_grid.predict(x_test)
#y_pred=best_serrch_grid.predict(x_test)

accuracy_score(y_test,y_pred)

