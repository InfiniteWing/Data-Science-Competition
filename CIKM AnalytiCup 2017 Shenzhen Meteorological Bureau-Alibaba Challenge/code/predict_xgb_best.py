import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder

PATH="../CIKM2017_train/"
testPath="../CIKM2017_testB/"
# read datasets
train = pd.read_csv(PATH+'train_pre_v2_kmeans.csv')
test = pd.read_csv(testPath+'testB_pre_v2_kmeans.csv')

# process columns, apply LabelEncoder to categorical features
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values) + list(test[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

# shape        
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))

from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
n_comp = 16

# PCA
pca = PCA(n_components=n_comp, random_state=0)
pca2_results_train = pca.fit_transform(train.drop(["label"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=0)
ica2_results_train = ica.fit_transform(train.drop(["label"], axis=1))
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=0)
grp_results_train = grp.fit_transform(train.drop(["label"], axis=1))
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=0)
srp_results_train = srp.fit_transform(train.drop(["label"], axis=1))
srp_results_test = srp.transform(test)

# Append decomposition components to datasets
for i in range(1, n_comp+1):
    train['pca_' + str(i)] = pca2_results_train[:,i-1]
    test['pca_' + str(i)] = pca2_results_test[:, i-1]
    
    train['ica_' + str(i)] = ica2_results_train[:,i-1]
    test['ica_' + str(i)] = ica2_results_test[:, i-1]

    train['grp_' + str(i)] = grp_results_train[:,i-1]
    test['grp_' + str(i)] = grp_results_test[:, i-1]
    
    train['srp_' + str(i)] = srp_results_train[:,i-1]
    test['srp_' + str(i)] = srp_results_test[:, i-1]


y_train = train["label"].values.astype(np.float32)
y_mean = np.mean(y_train)

# mmm, xgboost, loved by everyone ^-^
import xgboost as xgb

# prepare dict of params for xgboost to run with
xgb_params = {
    'n_trees': 100, 
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.5,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}

# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(train.drop('label', axis=1), y_train)
dtest = xgb.DMatrix(test)

# xgboost, cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
				   nfold=5,
                   num_boost_round=100, # increase to have better results (~700)
                   early_stopping_rounds=50,
                   verbose_eval=10, 
                   show_stdv=False
                  )

num_boost_rounds = len(cv_result)

# train model
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
pred = model.predict(dtest)
y_pred=[]
for i,predict in enumerate(pred):
	if(predict<0):
		y_pred.append(0)
	else:
		predict=predict*0.9+15.5*0.1
		y_pred.append(str(predict))
output = pd.DataFrame({'label': y_pred})
from datetime import datetime
output.to_csv('submission_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)