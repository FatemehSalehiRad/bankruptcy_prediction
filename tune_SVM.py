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
from sklearn.svm import SVC
#from sklearn.neural_network import MLPClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import GradientBoostingClassifier
#from xgboost import XGBClassifier
#from catboost import CatBoostClassifier 

#model=LogisticRegression()
model=SVC()
#model = MLPClassifier()
#model = RandomForestClassifier()
#model=GradientBoostingClassifier()
#model=XGBClassifier()
#model=CatBoostClassifier()

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=12)

model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred_train=model.predict(x_train)
print('Train accuracy',accuracy_score(y_train,y_pred_train))
print('Test accuracy',accuracy_score(y_test,y_pred))



# Randomized Search CV

from sklearn.model_selection import RandomizedSearchCV

#Regularization parameter
C=[int(x) for  x  in np.linspace(start=1,stop=100,num=100)]

#Specifies the kernel type to be used in the algorithm
kernel=['linear', 'poly', 'rbf', 'sigmoid']

#Degree of the polynomial kernel function (‘poly’)
degree=[int(x) for  x  in np.linspace(start=1,stop=10,num=10)]

#create the random grid
random_grid={'C'          :C,
             'kernel'     :kernel,
             'degree'     :degree, 
}

random_search=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=100,scoring='roc_auc',n_jobs=-1,cv=10)
random_search.fit(x_train,y_train)

random_search.best_params_

best_random_grid=random_search.best_estimator_

y_pred=best_random_grid.predict(x_test)

accuracy_score(y_test,y_pred)

# GridSearchCV

from sklearn.model_selection import GridSearchCV

param_grid={
    'kernel'       :[random_search.best_params_['kernel']],
    
    'C'            :[random_search.best_params_['C']-2,
                     random_search.best_params_['C']-1,
                     random_search.best_params_['C'],
                     random_search.best_params_['C']+1,
                     random_search.best_params_['C']+2],
    'degree'        :[random_search.best_params_['degree']-2,
                     random_search.best_params_['degree']-1,
                     random_search.best_params_['degree'],
                     random_search.best_params_['degree']+1,
                     random_search.best_params_['degree']+2],
   
}

grid_search=GridSearchCV(model,param_grid,scoring='roc_auc',n_jobs=-1,cv=10)
grid_search.fit(x_train,y_train)

best_search_grid=grid_search.best_estimator_

grid_search.best_params_

y_pred=best_search_grid.predict(x_test)

accuracy_score(y_test,y_pred)

# Beysian Optimization

from hyperopt import hp,fmin,tpe,STATUS_OK,Trials

space={'C': hp.quniform('C', 1, 20,1),
       'degree':hp.quniform('degree', 1,10, 1),
       'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
      }

def hyperparameter_tuning(space):
    model = SVC(C =space['C'],
                          degree = space['degree'],
                          kernel=space['kernel'],
                         )
    evaluation = [( x_train, y_train), ( x_test, y_test)]
    
    model.fit(x_train, y_train,
         )

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



