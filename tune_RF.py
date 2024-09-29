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
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)

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

random_search=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=100,scoring='roc_auc',n_jobs=-1,cv=10)
random_search.fit(x_train,y_train)

random_search.best_params_

best_random_grid=random_search.best_estimator_


y_pred=best_random_grid.predict(x_test)


accuracy_score(y_test,y_pred)

# GridSearchCV

random_search.best_params_

from sklearn.model_selection import GridSearchCV

param_grid={
    'criterion'         :[random_search.best_params_['criterion']],
    'max_depth'         :[random_search.best_params_['max_depth']],
    'max_features'      :[random_search.best_params_['max_features']],
    'min_samples_leaf'  :[random_search.best_params_['min_samples_leaf'],
                         random_search.best_params_['min_samples_leaf']+2,
                         random_search.best_params_['min_samples_leaf']+4],
    'min_samples_split' :[random_search.best_params_['min_samples_split']-2,
                          random_search.best_params_['min_samples_split']-1,
                          random_search.best_params_['min_samples_split'],
                          random_search.best_params_['min_samples_split']+1,
                          random_search.best_params_['min_samples_split']+2],
    'n_estimators'      :[random_search.best_params_['n_estimators']-200,
                          random_search.best_params_['n_estimators']-100,
                          random_search.best_params_['n_estimators'],
                          random_search.best_params_['n_estimators']+100,
                          random_search.best_params_['n_estimators']+200],
}

grid_search=GridSearchCV(model,param_grid,scoring='roc_auc',n_jobs=-1,cv=10)
grid_search.fit(x_train,y_train)

best_search_grid=grid_search.best_estimator_

y_pred=best_search_grid.predict(x_test)

accuracy_score(y_test,y_pred)

# Beysian Optimization

from hyperopt import hp,fmin,tpe,STATUS_OK,Trials

space={'criterion':hp.choice('criterion',['entropy','gini']),
       'max_depth' :hp.quniform('max_depth',10,1200,10),
       'max_features' :hp.choice('max_features' ,['auto','sqrt','log2',None]),
       'min_samples_leaf':hp.uniform('min_samples_leaf',0,0.5),
       'min_samples_split':hp.uniform('min_samples_split',0,1),
       'n_estimators' :hp.choice('n_estimators',[10,50,300,750,1200,1300,1500]),
}

def objective(space):
    model=RandomForestClassifier(criterion=space['criterion'],
                                  max_depth=space['max_depth'] ,
                                  max_features=space['max_features'],
                                  min_samples_leaf=space['min_samples_leaf'],
                                  min_samples_split=space['min_samples_split'],
                                   n_estimators=space['n_estimators'],
                                )
    
    accuracy=cross_val_score(model,x_train,y_train,cv=5).mean()
    
    return{'loss':-accuracy,'status':STATUS_OK}

from sklearn.model_selection import cross_val_score
trials=Trials()
best=fmin(fn=objective,
         space=space,
         algo=tpe.suggest,
         max_evals=80,
         trials=trials)
best

crit={0:'entropy',1:'gini'}
feat={0:'auto',1:'sqrt',2:'log2',3:None}
est={0:10,1:50,2:300,3:750,4:1200,5:1300,6:1500}

print(crit[best['criterion']])
print(feat[best['max_features']])
print(est[best['n_estimators']])

trainedforest=RandomForestClassifier(criterion=crit[best['criterion']],
                                  max_depth=best['max_depth'] ,
                                  max_features=feat[best['max_features']],
                                  min_samples_leaf=best['min_samples_leaf'],
                                  min_samples_split=best['min_samples_split'],
                                   n_estimators=est[best['n_estimators']],
                                ).fit(x_test,y_test)

predictionforest=trainedforest.predict(x_test)

accuracy_score(y_test,predictionforest)

