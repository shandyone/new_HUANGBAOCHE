# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
#import read_csv
import pywt  # python 小波变换的包

# 取数据
action_train = pd.read_csv("feature_process/action_process_train.csv", index_col=None)
action_test = pd.read_csv("feature_process/action_process_test.csv", index_col=None)
orderFuture_train = pd.read_csv("trainingset/orderFuture_train.csv", index_col=None)
orderFuture_test = pd.read_csv("test/orderFuture_test.csv", index_col=None)
userid_train = orderFuture_train['userid']
userid_test = orderFuture_test['userid']

actionTypetrain = {}
for i in userid_train:
    actionType_train = action_train[action_train['userid'] == i]['actionType'].tolist()
    actionTypetrain[i] = actionType_train

actionTypetest = {}
for i in userid_test:
    actionType_test = action_test[action_test['userid'] == i]['actionType'].tolist()
    actionTypetest[i] = actionType_test
print 'actionType okay'

data_train = actionTypetrain[100000002243]

class wavelet:
    def __init__(self, index_list, wavefunc='db4', lv=2, m=2, n=2, plot=True):# 默认小波函数为db4, 分解层数为4， 选出小波层数为1-4层
        self.index_list = index_list
        self.wavefunc = wavefunc
        self.lv = lv
        self.m = m
        self.n = n
        self.plot = plot

    def wt(self):  # 打包为函数，方便调节参数。  lv为分解层数；data为最后保存的dataframe便于作图；index_list为待处理序列；wavefunc为选取的小波函数；m,n则选择了进行阈值处理的小波系数层数
        # 分解
        coeff = pywt.wavedec(self.index_list, self.wavefunc, mode='sym', level=self.lv)  # 按 level 层分解，使用pywt包进行计算， cAn是尺度系数 cDn为小波系数
        sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0  # sgn函数
        # 去噪过程
        for i in range(self.m, self.n + 1):  # 选取小波系数层数为 m~n层，尺度系数不需要处理
            cD = coeff[i]
            for j in range(len(cD)):
                Tr = np.sqrt(2 * np.log(len(cD)))  # 计算阈值
                if cD[j] >= Tr:
                    coeff[i][j] = sgn(cD[j]) - Tr  # 向零收缩
                else:
                    coeff[i][j] = 0  # 低于阈值置零
        # 重构
        denoised_index = pywt.waverec(coeff, self.wavefunc)
        if self.plot == True:
            plt.figure()
            plt.plot(self.index_list, label='actual_data')
            plt.plot(denoised_index, label='denoised_data')
            plt.legend(loc='upper right')
            plt.savefig('pic/denoised.png')
            plt.show()
        return denoised_index

if __name__=='__main__':
    wt = wavelet(data_train).wt()