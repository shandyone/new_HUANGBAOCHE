# -*- coding:utf-8 -*-
from __future__ import division
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
userProfile = pd.read_csv('trainingset/userProfile_train.csv')
history = pd.read_csv('trainingset/orderHistory_train.csv')
feature_history = pd.read_csv('feature/feature_history_train.csv')
train = pd.read_csv('trainingset/orderFuture_train.csv')

feature_add_train = pd.read_csv('feature/feature_train_add.csv')
feature_add_test = pd.read_csv('feature/feature_test_add.csv')
merge_profile = pd.merge(userProfile, train)
#print merge
userid = merge_profile['userid']
userid_history = history['userid']
continent = history['continent']
country = history['country']

feature_add = feature_add_test
userid_add = feature_add['userid']
timeLast_add = feature_add['timeLast']
# label_add = feature_add['label']
#
# temp = {}
# for index, i in enumerate(timeLast_add):
#     temp[i] = label_add[index]
# sorted(temp.keys())
#
# time_dict = []
# label_dict = []
# for key in temp:
#     time_dict.append(key)
#     label_dict.append(temp[key])

# print time_dict
plt.figure()
#plt.plot(userid_add, timeLast_add, 'bo')
plt.scatter(userid_add, timeLast_add, marker='.')
plt.show()
'''
for i in userid:
    if int(feature_history[feature_history['userid']==i]['isHistory'])==1 and int(feature_history[feature_history['userid']==i]['isOrder'])==1:
        if int(train[train['userid']==i]['orderType'])==1:
            print i
#print merge[merge['province']=='河南']

check_province = '辽宁'
print len(merge_profile[merge_profile['province']==check_province ][merge_profile['orderType']==1])/len(merge_profile[merge_profile['province']==check_province])

check_continent_list = []
for i in continent:
    if i not in check_continent_list:
        check_continent_list.append(i)

#check_continent = '非洲'
for check_continent in check_continent_list:
    lenth = len(history[history['continent'] == check_continent])
    rate = len(history[history['continent'] == check_continent][history['orderType'] == 1])/len(history[history['continent'] == check_continent])
    print check_continent, lenth, rate

check_country_list = []
lenth_list = []
rate_list = []
for i in country:
    if i not in check_country_list:
        check_country_list.append(i)

#check_continent = '非洲'
for check_country in check_country_list:
    lenth = len(history[history['country'] == check_country])
    rate = len(history[history['country'] == check_country][history['orderType'] == 1])/len(history[history['country'] == check_country])
    lenth_list.append(lenth)
    rate_list.append(rate)
    print check_country, lenth, rate
'''


'''
comment = pd.read_csv('trainingset/userComment_train.csv')
train = pd.read_csv('trainingset/orderFuture_train.csv')
userid = train['userid']
type = train['orderType']

comment_userid = comment['userid']
rating = comment['rating']
tags = comment['tags']
commentsKeyWords = comment['commentsKeyWords']

comment_rating = {}
comment_tags = {}
comment_keywords = {}
for index, i in enumerate(comment_userid):
    comment_rating[i] = rating[index]
    comment_tags[i] = tags[index]
    comment_keywords[i] = commentsKeyWords[index]

count = 0
orderType = {}
for index, i in enumerate(userid):
    orderType[i] = type[index]

for i in comment_userid:
    if comment_rating[i] < 5 and orderType[i] == 1:
        print i, comment_rating[i], comment_tags[i], comment_keywords[i]
        count += 1
print count
'''


action_train = pd.read_csv("trainingset/action_train.csv", index_col=None)
action_test = pd.read_csv("test/action_test.csv", index_col=None)
order_train = pd.read_csv("trainingset/orderFuture_train.csv", index_col=None)
order_test = pd.read_csv("test/orderFuture_test.csv", index_col=None)
orderHistory_train = pd.read_csv("trainingset/orderHistory_train.csv", index_col=None)
orderHistory_test = pd.read_csv("test/orderHistory_test.csv", index_col=None)

# -------------------处理时间----------------------------
def date_prodcess(action, time):
    date = pd.to_datetime(action[time], unit='s')# format='%Y/%m/%d %H:%M:%S')
    year, month, day, hour, min = [], [], [], [], []
    for i in date:
        year.append(i.strftime('%Y'))
        month.append(i.strftime('%m'))
        day.append(i.strftime('%d'))
        hour.append(i.strftime('%H'))
        min.append(i.strftime('%M'))

    action['year'] = pd.DataFrame({'year': year})
    action['month'] = pd.DataFrame({'month': month})
    action['day'] = pd.DataFrame({'day': day})
    action['hour'] = pd.DataFrame({'hour': hour})
    action['min'] = pd.DataFrame({'min': min})
    return action

# -------------------处理用户id--------------------------
# 5填写 6提交 7下单 8确认 9支付
def encoder_userid(train):
    le = LabelEncoder()
    train['userid'] = le.fit_transform(train['userid'])
    return train

def get_dummies(train):
    dummy_label = ['year', 'month', 'day', 'hour']
    train = pd.get_dummies(train, prefix=dummy_label)
    return train

def drop_label(df):
    drop_label = ['actionTime','userid']
    return df.drop(drop_label, axis=1)

if __name__ =="__main__":
    action = date_prodcess(action_train, "actionTime")
    # train = pd.merge(action, order_train)  # 合并两个表

    # train = encoder_userid(train)
    # train = get_dummies(train)
    # train = drop_label(train)
    action.to_csv('feature_process/action_process_train.csv', index=False)

    action = date_prodcess(action_test, "actionTime")
    action.to_csv('feature_process/action_process_test.csv', index=False)

    action = date_prodcess(orderHistory_train, "orderTime")
    action.to_csv('feature_process/actionHistory_process_train.csv', index=False)

    action = date_prodcess(orderHistory_test, "orderTime")
    action.to_csv('feature_process/actionHistory_process_test.csv', index=False)