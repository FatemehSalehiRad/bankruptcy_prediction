# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 16:16:06 2022

@author: ST-01
"""

import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('D:/scale_data.csv')

x=df.iloc[:,:-1]
y=df['F40']

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier 

from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt


# Create the RFE object and rank each pixel
#model=LogisticRegression()
model = SVC(kernel='linear')
#model=MLPClassifier()
#model=RandomForestClassifier()
#model=GradientBoostingClassifier()
#model=XGBClassifier()
#model=CatBoostClassifier()


rfe = RFE(estimator=model, n_features_to_select=25, step=1)
fit=rfe.fit(x, y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

