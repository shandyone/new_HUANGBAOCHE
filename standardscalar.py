# -*- coding:utf-8 -*-
from __future__ import division
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

orderFuture_train = pd.read_csv("trainingset/orderFuture_train.csv", index_col=None)
train_add = pd.read_csv('feature/feature_train_add.csv')
test_add = pd.read_csv('feature/feature_test_add.csv')
# orderFuture_test = pd.read_csv("test/orderFuture_test.csv", index_col=None)
# userid_train = orderFuture_train['userid'].tolist()
# userid_test = orderFuture_test['userid'].tolist()
# feature_train = pd.read_csv('feature/feature_train_add.csv')
# feature_test = pd.read_csv('feature/feature_test_add.csv')
train = pd.read_csv('feature/REPOTR_01212310.csv')
test = pd.read_csv('feature/REPOTT_01212310.csv')
train['label'] = orderFuture_train['orderType']
train['userid_kmeans'] = train_add['label_pred']
test['userid_kmeans'] = test_add['label_pred']
'''
#columns_train = feature_train.columns
columns_test = feature_test.columns
columns_train = [x for x in feature_train.columns if x != 'label']

print columns_train
feature_array_train = []
for i in columns_train:
    feature_array_train.append(feature_train[i].tolist())
scaler_train = preprocessing.MinMaxScaler()
train_scaled = scaler_train.fit_transform(feature_array_train)

feature_train_scaled = pd.DataFrame(columns=columns_train, index=None)
for index, i in enumerate(train_scaled):
    feature_train_scaled[columns_train[index]] = i
feature_train_scaled['label'] = feature_train['label']
feature_train_scaled.to_csv('feature/train_scaled.csv',index=None)


feature_array_test = []
for i in columns_test:
    feature_array_test.append(feature_test[i].tolist())
scaler_test = preprocessing.MinMaxScaler()
test_scaled = scaler_test.fit_transform(feature_array_test)

feature_test_scaled = pd.DataFrame(columns=feature_test.columns, index=None)
for index, i in enumerate(test_scaled):
    feature_test_scaled[columns_test[index]] = i
'''
train.to_csv('feature/REPOTR_01212310.csv', index=None)
test.to_csv('feature/REPOTT_01212310.csv', index=None)