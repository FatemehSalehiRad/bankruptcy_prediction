# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 15:50:51 2021

@author: ST-01
"""

import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('D:/scale_data.csv')

x=df.iloc[:,:-1]
y=df['F40']

import seaborn as sns
corr=df.iloc[:,:-1].corr()
top_features=corr.index
plt.figure(figsize=(20,20))
sns.heatmap(df[top_features].corr(),annot=True)

threshold=0.8

# find and remove correlated features

def correlation(dataset,threshold):
    col_corr=set()#set of all the names of correlated columns
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])> threshold: #we are interested in absolute coeff value
                colname=corr_matrix.columns[i] #getting the name of column
                col_corr.add(colname)
    return col_corr 

# correlation(df.iloc[:,:-1],threshold)


# extra tree classifier

from sklearn.ensemble import ExtraTreesClassifier
model=ExtraTreesClassifier()
model.fit(x,y)

ranked_features=pd.Series(model.feature_importances_,index=x.columns)
ranked_features.nlargest(20).plot(kind='barh')
plt.show()

#TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.manifold import TSNE


df=pd.read_csv('D:/scale_data.csv')
x=df.iloc[:,0:39].values
y=df.iloc[:,39]

tsne= TSNE(n_components=20, learning_rate=200,method='exact',
                  init='random')

x=tsne.fit_transform(x)