# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 12

train = pd.read_csv('feature/feature_history_train.csv')
test = pd.read_csv('feature/feature_history_test.csv')

target = 'label'
userid = 'userid'

predictors = [x for x in train.columns if x not in [target, userid]]
X_train, X_test, y_train, y_test = train_test_split(train[predictors], train[target], test_size=0.2, random_state=43)
print "read_csv is okay!"

def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
        print cvresult

    alg.fit(X_train, y_train, eval_metric='auc')

    dtrain_predictions = alg.predict(X_test)
    dtrain_predprob = alg.predict_proba(X_test)[:, -1]

    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(y_test, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(y_test, dtrain_predprob)

    plt.figure()
    feat_imp = pd.Series(alg._Booster.get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()


xgb1 = XGBClassifier(
 tree_method='gpu_exact',
 learning_rate=0.1,
 n_estimators=1154,
 max_depth=5,
 min_child_weight=5,
 gamma=0.1,
 subsample=0.1,
 colsample_bytree=0.8,
 #reg_alpha=1,
 reg_lambda=10,
 objective='binary:logistic',
 eval_metric='auc',
 nthread=12,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)