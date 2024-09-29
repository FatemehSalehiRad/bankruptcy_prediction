# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 16:07:13 2021

@author: ST-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data=pd.read_csv('D:/newdata.csv')
data1=pd.read_csv('D:/newdata1.csv')
exp=pd.read_csv('D:/example.csv')

plt.hist(exp.v11,bins=9,color='red')
plt.hist(exp.v12,bins=13,color='blue')
plt.grid()
plt.legend(['v','s'],loc='best')
plt.ylabel('Frequency')
plt.xlabel('F14')