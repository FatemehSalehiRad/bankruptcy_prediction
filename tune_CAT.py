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
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import GradientBoostingClassifier
#from xgboost import XGBClassifier
from catboost import CatBoostClassifier 

#model=LogisticRegression()
#model=SVC()
#model = MLPClassifier()
#model = RandomForestClassifier()
#model=GradientBoostingClassifier()
#model=XGBClassifier()
model=CatBoostClassifier()

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=12)

model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred_train=model.predict(x_train)
print('Train accuracy',accuracy_score(y_train,y_pred_train))
print('Test accuracy',accuracy_score(y_test,y_pred))


from sklearn.model_selection import RandomizedSearchCV
#Number of trees in a RandomForest
n_estimators=[int(x) for  x  in np.linspace(start=200,stop=2000,num=10)]

#Max Number of levels in a tree
max_depth=[int(x) for x in np.linspace(10,1000,10)]

#The fraction of observations to be selected for each tree
subsample=[0.1,0.2,0.3,0.5,0.7,1]

#Learning rate shrinks the contribution of each tree 
learning_rate=[0.05,0.1,0.15,0.2,0.25,0.3]

#Subsample ratio of columns when constructing each tree.
colsample_bytree=[0.2,0.3,0.4,0.5,0.6,0.7]

#Coefficient at the L2 regularization term of the cost function
l2_leaf_reg=[0.1,0.5,1,2,10,50,100]

#create the random grid
random_grid={'n_estimators'     :n_estimators,
             'max_depth'        :max_depth,
             'subsample'        :subsample,
             'learning_rate'    :learning_rate,
             'l2_leaf_reg'      :l2_leaf_reg,
             
}

random_search=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=50,scoring='roc_auc',n_jobs=-1,cv=5)
random_search.fit(x_train,y_train)

random_search.best_params_

best_random_grid=random_search.best_estimator_

y_pred=best_random_grid.predict(x_test)

accuracy_score(y_test,y_pred)

# GridSearchCV

from sklearn.model_selection import GridSearchCV

param_grid={
    'max_depth'          :[random_search.best_params_['max_depth']],
    'colsample_bytree'   :[random_search.best_params_['colsample_bytree']-0.2,
                          random_search.best_params_['colsample_bytree']-0.1,
                          random_search.best_params_['colsample_bytree'],
                          random_search.best_params_['colsample_bytree']+0.2,
                          random_search.best_params_['colsample_bytree']+0.1],
    'n_estimators'       :[random_search.best_params_['n_estimators']-200,
                          random_search.best_params_['n_estimators']-100,
                          random_search.best_params_['n_estimators'],
                          random_search.best_params_['n_estimators']+100,
                          random_search.best_params_['n_estimators']+200],
    'learning_rate'      :[random_search.best_params_['learning_rate']-0.1,
                          random_search.best_params_['learning_rate']-0.05,
                          random_search.best_params_['learning_rate'],
                          random_search.best_params_['learning_rate']+0.1,
                          random_search.best_params_['learning_rate']+0.05],
    'subsample'          :[random_search.best_params_['subsample']-0.2,
                          random_search.best_params_['subsample']-0.1,
                          random_search.best_params_['subsample'],
                          random_search.best_params_['subsample']+0.2,
                          random_search.best_params_['subsample']+0.1],
    'l2_leaf_reg'         :[random_search.best_params_['l2_leaf_reg']],
}

grid_search=GridSearchCV(model,param_grid,scoring='roc_auc',n_jobs=-1,cv=5)
grid_search.fit(x_train,y_train)

best_search_grid=grid_search.best_estimator_

y_pred=best_search_grid.predict(x_test)

accuracy_score(y_test,y_pred)

# Baysian Optimizatin

from hyperopt import hp,fmin,tpe,STATUS_OK,Trials

space={ 'learning_rate': hp.uniform('learning_rate', 0.01, 0.8),
    'max_depth': hp.quniform('max_depth', 2, 10, 1),
    'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1.0),
    'bagging_temperature': hp.uniform('bagging_temperature', 0.0, 100),
    'random_strength': hp.uniform('random_strength', 0.0, 100),
    'scale_pos_weight': hp.uniform('scale_pos_weight', 1.0, 16.0)
}

def hyperparameter_tuning(space):
    model = CatBoostClassifier(colsample_bylevel =space['colsample_bylevel'],
                          max_depth = int(space['max_depth']),
                          learning_rate=space['learning_rate'],
                          bagging_temperature=space['bagging_temperature'],
                          random_strength=space['random_strength'],
                          scale_pos_weight=space['scale_pos_weight'],     
                         )
    evaluation = [( x_train, y_train), ( x_test, y_test)]
    
    model.fit(x_train, y_train,
            eval_set=evaluation,
            early_stopping_rounds=10,verbose=False)

    pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, pred)
    print ("SCORE:", accuracy)
    #change the metric if you like
    return {'loss': -accuracy, 'status': STATUS_OK, 'model': model}

trials = Trials()
best = fmin(fn=hyperparameter_tuning,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print (best)

