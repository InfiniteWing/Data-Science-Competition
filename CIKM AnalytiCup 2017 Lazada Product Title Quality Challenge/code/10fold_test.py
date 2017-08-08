import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
### Regressor
import xgboost as xgb
# read datasets
X = pd.read_csv('../training/data_train_pre_word2vec.csv',encoding = 'utf-8',quotechar='"')
X.fillna(0)
# process columns, apply LabelEncoder to categorical features
for c in X.columns:
    if(c=="clarity" or c=="conciseness"):
        continue
    if X[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(X[c].values)) 
        X[c] = lbl.transform(list(X[c].values))

# shape      
print('Shape train: {}'.format(X.shape))


TYPES=['clarity','conciseness']
rounds={}
rounds['clarity']=150
rounds['conciseness']=800
for TYPE in TYPES:
    print(TYPE)  
    
    label_col=['conciseness','clarity']
    drop_col=['sku_id','price']
    y = X[TYPE]
    y = np.array(y,np.int32)
    X1=np.array(X.drop(drop_col+label_col, axis=1))
    kf=KFold(n_splits=10, random_state=0, shuffle=True)
    labels=[]
    preds=[]
    round=0
    for train_index, test_index in kf.split(X1):
        round+=1
        print("Train round {}".format(round))
        #print("TRAIN:", train_index, "TEST:", test_index)
        train, test = X1[train_index], X1[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
    
        y_mean = np.mean(y_train)


        # prepare dict of params for xgboost to run with
        xgb_params = {
            'n_trees': 520, 
            'eta': 0.05,
            'max_depth': 5,
            'subsample': 0.9,
            'objective': 'reg:logistic',
            'eval_metric': 'rmse',
            'base_score': y_mean, # base prediction = mean(target)
            'silent': 1
        }
        #print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))
        dtrain = xgb.DMatrix(train, y_train)
        dtest = xgb.DMatrix(test, y_test)
        # xgboost, cross-validation
        '''
        cv_result = xgb.cv(xgb_params, 
                           dtrain, 
                           num_boost_round=10000, # increase to have better results (~700)
                           early_stopping_rounds=50,
                           show_stdv=False
                          )
        '''
        num_boost_rounds = 150#len(cv_result)
        # train model
        model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
        label=list(dtest.get_label())
        pred=list(model.predict(dtest))
        
        labels+=label
        preds+=pred
        RMSE = mean_squared_error(label, pred)**0.5
        print("Valid RMSE = {}".format(RMSE))
    RMSE = mean_squared_error(labels, preds)**0.5
    print("Avg RMSE = {}".format(RMSE))
    