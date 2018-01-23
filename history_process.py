# -*- coding:utf-8 -*-
from __future__ import division
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

orderFuture_train = pd.read_csv("trainingset/orderFuture_train.csv", index_col=None)
orderFuture_test = pd.read_csv("test/orderFuture_test.csv", index_col=None)
history_train = pd.read_csv('trainingset/orderHistory_train.csv')
history_test = pd.read_csv('test/orderHistory_test.csv')
history_train_userid = history_train['userid']
history_test_userid = history_test['userid']

userid_train = orderFuture_train['userid']
userid_test = orderFuture_test['userid']
label_train = orderFuture_train['orderType']
feature_train = pd.DataFrame({
    'userid': userid_train,
    'label': label_train
}, columns = ['userid', 'label'])
feature_test = pd.DataFrame({'userid': userid_test})
print 'read csv okay'

#-------------------------history----------------------
def isHistory(train_test, history):
    isHistory = []
    isOrder = []
    for i in train_test:
        if i in history['userid'].tolist():
            isHistory.append(1)
            if history[history['userid']==i][history['orderType'].isin(['1'])].empty == True:
                isOrder.append(1)
            else:
                isOrder.append(0)
        else:
            isHistory.append(0)
            isOrder.append(2)
    return isHistory, isOrder
isHistory_train, isOrder_train = isHistory(userid_train, history_train)
isHistory_test, isOrder_test = isHistory(userid_test, history_test)
feature_train['isHistory'] = isHistory_train
feature_train['isOrder'] = isOrder_train
feature_test['isHistory'] = isHistory_test
feature_test['isOrder'] = isOrder_test
print 'history okay'

# def getDummies(train_test):
#     label = ['isHistory', 'isOrder']
#     temp1 = pd.get_dummies(train_test['isHistory'], prefix='isHistory')
#     temp2 = pd.get_dummies(train_test['isOrder'], prefix='isOrder')
#     train_test = pd.DataFrame(pd.concat([train_test, temp1], axis=1))
#     train_test = pd.DataFrame(pd.concat([train_test, temp2], axis=1))
#     train_test = pd.DataFrame(train_test.drop(label, axis=1))
#     return train_test
#
# feature_train = getDummies(feature_train)
# feature_test = getDummies(feature_test)
print 'dummies okay'

feature_train.to_csv('feature/feature_history_train.csv', index=None)
feature_test.to_csv('feature/feature_history_test.csv', index=None)