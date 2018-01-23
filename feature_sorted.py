# -*- coding:utf-8 -*-
from __future__ import division
import pandas as pd
import numpy as np

# train = pd.read_csv('feature/feature_train_add.csv')
# test = pd.read_csv('feature/feature_test_add.csv')
train = pd.read_csv('feature/REPOTR_01212310.csv')
test = pd.read_csv('feature/REPOTT_01212310.csv')

target = 'label'
predictors_train = [x for x in train.columns if x not in [target]]
predictors_test = [x for x in test.columns if x not in [target]]

def sorted_feature(feature, predictors):
    for i in predictors:
        key = str(i) + 'sorted'
        #value = np.argsort(feature[i])
        value = feature[i].rank(method='max')
        feature[key] = value
    return feature

feature_train = sorted_feature(train, predictors_train)
print 'train okay~'
feature_test = sorted_feature(test, predictors_test)
print 'test okay~'
feature_train.to_csv('feature/feature_train_sorted.csv', index=None)
feature_test.to_csv('feature/feature_test_sorted.csv', index=None)