# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 09:36:13 2021

@author: ST-01
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

data=pd.read_csv('D:/data.csv')
data1=pd.read_csv('D:/data1.csv')
data.drop('Year',axis=1,inplace=True)
data.drop('Name',axis=1,inplace=True)


imp=SimpleImputer(missing_values=np.nan,strategy='mean')
imp.fit(data)
imp.fit(data1)
newdata= imp.transform(data)
newdata1= imp.transform(data1)
