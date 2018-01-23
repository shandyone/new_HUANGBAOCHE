# -*- coding:utf-8 -*-
from __future__ import division
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

feature_score = pd.read_csv('feature_score/feature_score.csv')
feature_train = pd.read_csv('feature/feature_train.csv')
feature_test = pd.read_csv('feature/feature_test.csv')
print 'read csv okay'

# ---------------poly--------------------------------------
feature_list = feature_score[(feature_score.score > 200)].feature.tolist()
train_combine = feature_train[feature_list]
test_combine = feature_test[feature_list]

poly = PolynomialFeatures(interaction_only=True)
train_combine = poly.fit_transform(train_combine)
df = pd.DataFrame(train_combine.reshape(40307, -1))
df.interpolate(axis=1).values.reshape(train_combine.shape)
#df[df.columns] = MinMaxScaler().fit_transform(df[df.columns])
train_combine = pd.concat([feature_train, df], axis=1)
feature_train = pd.DataFrame(train_combine)
print 'train poly okay'

poly = PolynomialFeatures(interaction_only=True)
test_combine = poly.fit_transform(test_combine)
df = pd.DataFrame(test_combine.reshape(10076, -1))
df.interpolate(axis=1).values.reshape(test_combine.shape)
#df[df.columns] = MinMaxScaler().fit_transform(df[df.columns])
test_combine = pd.concat([feature_test, df], axis=1)
feature_test = pd.DataFrame(test_combine)
print 'test poly okay'

feature_train.to_csv('feature/feature_train_poly.csv', index=None)
feature_test.to_csv('feature/feature_test_poly.csv', index=None)