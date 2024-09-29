# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 10:09:20 2021

@author: ST-01
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import scale

data=pd.read_csv('D:/newdata.csv')
data1=pd.read_csv('D:/newdata1.csv')

df_data=scale(data)
df_data1=scale(data1)

scale_data=pd.DataFrame(df_data,
                        index=data.index,
                        columns=data.columns)
scale_data1=pd.DataFrame(df_data1,
                         index=data1.index,
                         columns=data1.columns)