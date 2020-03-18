# -*- coding: utf-8 -*-
#
# 熵、经验熵、经验条件熵、信息增益、信息增益比的定义
#
#

import math
from data_type import *
# 计算熵
def entropy(p):
    e = 0
    for i in p:
        if i != 0:
            e += (i*math.log2(i))
    return e

def empirical_entropy(dataset):
    "数据集的在标签上的经验熵"
    # 类别数
    categories = len(dataset.label_type())
    p = []
    for c in range(categories):
        count = 0
        for i in range(len(dataset.sample_list)):
            if dataset.getLabel(i).value == c:
                count+=1
        p.append(count/len(dataset.sample_list))
    return (-1)*entropy(p)

# 参数 feature_type 为特征的实例 借以表明条件熵所在的特征
def conditional_entropy(dataset, feature_type):
    "经验条件熵"
    #print(type(feature_type))
    # 特征可取值个数
    categories = len(feature_type)
    p = []
    h = []
    for c in range(categories):
        tmp_dataset = DataSet()
        count = 0
        for i in range(len(dataset.sample_list)):
            if dataset.feature_value(i,feature_type(0)).value == c:
                count += 1
                tmp_dataset.addSample(dataset.getSample(i))
        h.append(empirical_entropy(tmp_dataset))
        p.append(count/dataset.size())

    res = 0
    for i in range(len(p)):
        res += (p[i]*h[i])

    return res


def gain(dataset, feature_type):
    "特征feature_type对数据集dataset的信息增益"
    return empirical_entropy(dataset)-conditional_entropy(dataset,feature_type)

def gainR(dataset, feature_type):
    return gain(dataset,feature_type)/empirical_entropy(dataset)