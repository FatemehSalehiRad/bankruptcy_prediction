# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 10:52:44 2021

@author: ST-01
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

df=pd.read_csv('D:/scale_data.csv')


x=df.drop(['F40'],axis=1)
y=df['F40']

x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=5)

kernel=['linear','poly','sigmoid','rbf']


#for i in kernel:
  #  svclassifier=SVC(kernel=i)
   # svclassifier.fit(x_train, y_train)
    #y_pred=svclassifier.predict(x_test)
    #print('for kernel:',i)
    #print('accuracy is:',metrics.accuracy_score(y_test,y_pred))
    
   #for i in range(1,10):
   # svclassifier=SVC(kernel='poly',degree=i,C=1)
   # svclassifier.fit(x_train, y_train)
   # y_pred=svclassifier.predict(x_test)
 
   # print('for degree:',i)
   # print('accuracy is:',metrics.f1_score(y_test,y_pred))
    
   
    #Applying gridsearch
from sklearn.model_selection import GridSearchCV

parameter=[{'C':[0.1,1,100,1000],'kernel':['poly'],'degree':[1,2,3,4,5,6]},
           {'C':[0.1,1,100,1000],'kernel':['linear']},
           {'C':[0.1,1,100,1000],'kernel':['rbf'],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]
grid=GridSearchCV(SVC(),parameter,cv=10,n_jobs=-1)
grid=grid.fit(x_train,y_train)

grid.best_params_
    
    