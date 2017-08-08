import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
### Regressor
import xgboost as xgb
# read datasets
train = pd.read_csv('../training/data_train_pre_word2vec.csv',encoding = 'utf-8',quotechar='"')
train.fillna(0)
# process columns, apply LabelEncoder to categorical features
for c in train.columns:
    if(c=="clarity" or c=="conciseness"):
        continue
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values)) 
        train[c] = lbl.transform(list(train[c].values))

# shape      
print('Shape train: {}'.format(train.shape))


TYPES=['conciseness','clarity']
for TYPE in TYPES:
    print(TYPE)     
    label_col=['conciseness','clarity']
    drop_col=['sku_id','price']

    y_train = train[TYPE]
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

    dtrain = xgb.DMatrix(train.drop(drop_col+label_col, axis=1), y_train)
    # xgboost, cross-validation

    cv_result = xgb.cv(xgb_params, 
                       dtrain, 
                       num_boost_round=10000, # increase to have better results (~700)
                       early_stopping_rounds=50,
                       verbose_eval=10, 
                       show_stdv=False
                      )

    num_boost_rounds = len(cv_result)
    print('num_boost_rounds=' + str(num_boost_rounds))

    # train model
    model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

    #labels=dtrain.get_label()
    #predicts=model.predict(dtrain)
    #RMSE = mean_squared_error(labels, predicts)**0.5
    #print(RMSE)

    # make predictions and save results
    