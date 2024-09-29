# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 15:10:44 2021

@author: ST-01
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.manifold import TSNE


df=pd.read_csv('D:/scale_data.csv')
x=df.drop(['F40'],axis=1)
y=df['F40']

tsne= TSNE(n_components=2, learning_rate=200,method='exact',
                  init='random')

x=tsne.fit_transform(x)