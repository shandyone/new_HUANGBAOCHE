# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics   #Additional     scklearn functions
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 12

train = pd.read_csv('feature/feature_train_sorted.csv')
test = pd.read_csv('feature/feature_test_sorted.csv')
# train = pd.read_csv('feature_score/feature_history_train.csv')
# test = pd.read_csv('feature_score/feature_history_test.csv')
# train = pd.read_csv('feature/feature_train_add.csv')
# test = pd.read_csv('feature/feature_test_add.csv')
# train = pd.read_csv('feature/REPOTR_01212310.csv')
# test = pd.read_csv('feature/REPOTT_01212310.csv')
# train = pd.read_csv('feature/feature_train.csv')
# test = pd.read_csv('feature/feature_test.csv')
feature_score = pd.read_csv('feature_score/feature_score_poly.csv')

print len(train)
print len(train.columns)

TestResult = pd.read_csv('Result/test5.csv')
print len(test)
print len(test.columns)

target = 'label'
userid = 'userid'
predictors = [x for x in train.columns if x not in [target]]
#predictors = feature_score[(feature_score.score > 9)].feature.tolist()

X_train, X_test, y_train, y_test = train_test_split(train[predictors], train[target], test_size=0.1, random_state=43)

print "read_csv is okay!"

def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
        print cvresult

    #Fit the algorithm on the data
    # alg.fit(dtrain[predictors], dtrain[target],eval_metric='auc')
    alg.fit(X_train, y_train, eval_metric='auc')

    #Predict training set:
    # dtrain_predictions = alg.predict(dtrain[predictors])
    # dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]
    dtrain_predictions = alg.predict(X_test)
    dtrain_predprob = alg.predict_proba(X_test)[:, -1]

    #Print model report:
    print "\nModel Report"
    # print "Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions)
    print "Accuracy : %.4g" % metrics.accuracy_score(y_test, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(y_test, dtrain_predprob)
    # print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob)

    # select_model = SelectFromModel(alg, prefit=True)
    # X_new = select_model.transform(train[predictors].as_matrix())
    # Y_new = select_model.transform(test[predictors].as_matrix())
    # print len(X_new.columns)
    # print ('select okay')
    # X_new.to_csv('feature/feature_train.csv', index=None)
    # Y_new.to_csv('feature/feature_test.csv', index=None)

    feature_score = alg._Booster.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x:x[1], reverse=True)
    fs = []
    for (key, value) in feature_score:
        fs.append("{0},{1}\n".format(key, value))

    with open('feature_score/feature_score.csv', 'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)

    plt.figure()
    #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp = pd.Series(alg._Booster.get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.savefig('pic/xgboost.png')
    plt.show()


xgb1 = XGBClassifier(
 tree_method='gpu_exact',
 learning_rate=0.05,
 n_estimators=1009,
 max_depth=5,
 min_child_weight=8,
 gamma=0.1,
 subsample=0.1,
 colsample_bytree=0.8,
 #reg_alpha=1,
 reg_lambda=40,
 objective='binary:logistic',
 eval_metric='auc',
 nthread=12,
 scale_pos_weight=1,
 seed=27
 )
modelfit(xgb1, train, predictors)
# n_samples = train[predictors].shape[0]
# cv = ShuffleSplit(n_samples, test_size=0.3, random_state=0)
# cv_score = cross_val_score(xgb1, train[predictors], train[target], cv=10, scoring='roc_auc')
# print cv_score
# print np.mean(cv_score)
#
# xgb1.fit(train[predictors].as_matrix(), train[target].as_matrix())
# print 'fit success!'
#
# y_predict = xgb1.predict_proba(test[predictors].as_matrix())[:,-1]
# TestResult_label = [0 if val < 0.5 else 1 for val in TestResult['orderType'].values]
# print ("TestResult---->score_acu:", round(0.96810 *metrics.roc_auc_score(TestResult_label, y_predict), 5))
#
#
# submission = pd.DataFrame({
#     "orderType" : y_predict,
#     "userid" : test["userid"]
# }, columns = ['userid', 'orderType'])
# submission.to_csv('submission.csv', index=False)

# random_state = 5 1167 0.964330

'''
param_test1 = {
 'max_depth': range(3, 10, 2),
 'min_child_weight': range(1, 6, 2)
}
gsearch1 = GridSearchCV(estimator=XGBClassifier(tree_method='gpu_exact', learning_rate =0.1, n_estimators=167, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8,             colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4,     scale_pos_weight=1, seed=27),
 param_grid = param_test1,     scoring='roc_auc',n_jobs=4,iid=False, cv=5, verbose=10)
gsearch1.fit(train[predictors],train[target])
print gsearch1.cv_results_, gsearch1.best_params_,     gsearch1.best_score_
'''

'''
param_test2 = {
 'max_depth':[4,5,6],
 'min_child_weight':[4,5,6]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier(     learning_rate=0.1, n_estimators=91, max_depth=5,
 min_child_weight=5, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5,  verbose=10)
gsearch2.fit(train[predictors],train[target])
print gsearch2.cv_results_, gsearch2.best_params_,     gsearch2.best_score_
'''

'''
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=319, max_depth=5, min_child_weight=5, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5, verbose=10)

gsearch3.fit(train[predictors],train[target])
print gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_
'''
'''

param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}

gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=70, max_depth=7, min_child_weight=5, gamma=0.1, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5, verbose=10)

gsearch4.fit(train[predictors],train[target])
print gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_
'''
'''
param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier(  tree_method='gpu_hist',
 n_gpus=-1,learning_rate =0.1, n_estimators=70, max_depth=7, min_child_weight=5, gamma=0.1, subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', nthread=4, scale_pos_weight=1,seed=27), param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5, verbose=10)

gsearch6.fit(train[predictors],train[target])
print gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_
'''

#
# xgb1 = XGBClassifier(
#  tree_method='gpu_exact',
#  learning_rate=0.05,
#  n_estimators=1228,
#  max_depth=5,
#  min_child_weight=8,
#  gamma=0.1,
#  subsample=0.1,
#  colsample_bytree=0.8,
#  #reg_alpha=1,
#  reg_lambda=40,
#  objective='binary:logistic',
#  eval_metric='auc',
#  nthread=12,
#  scale_pos_weight=1,
#  seed=27
#  )

