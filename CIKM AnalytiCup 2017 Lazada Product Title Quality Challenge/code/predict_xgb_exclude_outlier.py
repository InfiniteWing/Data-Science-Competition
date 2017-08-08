import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
### Regressor
import xgboost as xgb
TYPE='conciseness'
# read datasets
train = pd.read_csv('../training/data_train_pre.csv',encoding = 'utf-8',quotechar='"')
train.fillna(0)
test = pd.read_csv('../validation/data_valid_pre.csv')

# process columns, apply LabelEncoder to categorical features
for c in train.columns:
    if(c=="clarity" or c=="conciseness"):
        continue
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values) + list(test[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

# shape        
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))


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

# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(train.drop(drop_col+label_col, axis=1), y_train)
dtest = xgb.DMatrix(test.drop(drop_col,axis=1))
# xgboost, cross-validation
'''
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=10000, # increase to have better results (~700)
                   early_stopping_rounds=50,
                   verbose_eval=10, 
                   show_stdv=False
                  )

num_boost_rounds = len(cv_result)
print('num_boost_rounds=' + str(num_boost_rounds))
'''
num_boost_rounds=600#118

# train model
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)


labels=dtrain.get_label()
predicts=model.predict(dtrain)
RMSE = mean_squared_error(labels, predicts)**0.5
print(RMSE)
drop_outlier_rows=[]
for i,pred in enumerate(predicts):
    label=labels[i]
    dif=abs(pred-label)
    if(dif>0.85):
        drop_outlier_rows.append(i)

train=train.drop(train.index[drop_outlier_rows])
y_train = train[TYPE]
y_mean = np.mean(y_train)

# shape        
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))
# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(train.drop(drop_col+label_col, axis=1), y_train)
dtest = xgb.DMatrix(test.drop(drop_col,axis=1))
# xgboost, cross-validation
'''
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=10000, # increase to have better results (~700)
                   early_stopping_rounds=50,
                   verbose_eval=10, 
                   show_stdv=False
                  )

num_boost_rounds = len(cv_result)
print('num_boost_rounds=' + str(num_boost_rounds))
'''
num_boost_rounds=500

# train model
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

labels=dtrain.get_label()
predicts=model.predict(dtrain)
RMSE = mean_squared_error(labels, predicts)**0.5
print(RMSE)

# make predictions and save results
preds = model.predict(dtest)
y_pred=[]
for pred in preds:
    #if(pred<0.1):
    #    pred=0
    y_pred.append(pred)
from datetime import datetime
output = pd.DataFrame({TYPE: y_pred})
output.to_csv('{}_valid.predict'.format(TYPE),index=False,header=False)
