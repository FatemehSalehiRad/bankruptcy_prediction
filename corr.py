# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 12:00:22 2021

@author: ST-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

dataset = pd.read_csv('D:/scale_data.csv')



corr = dataset.corr()
plt.figure(figsize=(25,25))
sb.heatmap(corr, annot=True, xticklabels=corr.columns.values, yticklabels=corr.columns.values,cmap='viridis')
