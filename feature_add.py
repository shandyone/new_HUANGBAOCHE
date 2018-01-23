# -*- coding:utf-8 -*-
from __future__ import division
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

train_demon = pd.read_csv('feature_process/TR_01031509.csv')
test_demon = pd.read_csv('feature_process/TT_01031509.csv')
# feature_train = pd.read_csv('feature/feature_train_add.csv')
# feature_test = pd.read_csv('feature/feature_test_add.csv')
feature_train = pd.read_csv('feature/feature_train.csv')
feature_test = pd.read_csv('feature/feature_test.csv')
orderFuture_train = pd.read_csv("trainingset/orderFuture_train.csv", index_col=None)
orderFuture_test = pd.read_csv("test/orderFuture_test.csv", index_col=None)
userid_train = orderFuture_train['userid']
userid_test = orderFuture_test['userid']

timeTrans42_train = train_demon['timeTrans42'].tolist()
feature_train['timeTrans42'] = timeTrans42_train

timeTrans42_test = test_demon['timeTrans42'].tolist()
feature_test['timeTrans42'] = timeTrans42_test

province_train = train_demon['province'].tolist()
feature_train['province'] = province_train

province_test = test_demon['province'].tolist()
feature_test['province'] = province_test

# ---------------K-means-------------------
''''''
userid_reshape_train = np.array(userid_train).reshape(-1,1)
userid_reshape_test = np.array(userid_test).reshape(-1,1)
estimator_train = KMeans(n_clusters=10, init='k-means++', random_state=2)
estimator_train.fit(userid_reshape_train)

label_pred_train = estimator_train.labels_
label_pred_test = estimator_train.predict(userid_reshape_test)

print label_pred_train
print label_pred_test
# temp_label_train = pd.get_dummies(label_pred_train, prefix='labelPred')
# temp_label_test = pd.get_dummies(label_pred_test, prefix='labelPred')
# feature_train = pd.DataFrame(pd.concat([feature_train, temp_label_train], axis=1))
# feature_test = pd.DataFrame(pd.concat([feature_test, temp_label_test], axis=1))

# feature_train = pd.DataFrame(feature_train.drop('label_pred', axis=1))
# feature_test = pd.DataFrame(feature_test.drop('label_pred', axis=1))

feature_train['label_pred'] = label_pred_train
feature_test['label_pred'] = label_pred_test
print 'k-means okay~'


feature_train.to_csv('feature/feature_train_add.csv', index=None)
feature_test.to_csv('feature/feature_test_add.csv', index=None)