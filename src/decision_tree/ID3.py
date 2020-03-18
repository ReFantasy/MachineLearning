# -*- coding: utf-8 -*-
#
# ID3算法以及决策树的定义
#
#

from data_type import *
from entropy import *

def is_same_kind(dataset):
    "判断数据集中的样本是否属于同一类"
    same = 1
    label = dataset.getLabel(0)
    for i in range(1,dataset.size()):
        if dataset.getLabel(i).value != dataset.getLabel(i-1).value:
            same = 0
    return [same,label]


class Node:
    def __init__(self, D=DataSet(), A = [], el =0, label=-1):
        # 划分到该节点下的数据
        self.D = D
        # 特征集
        self.A = A
        # 终止阈值
        self.el =el

        # 节点属于的类别
        self.label = label
        # 节点的子树
        self.child = []

        # 特征值到节点的映射
        self.dict = {}

        self.leaf = 0

        # 该节点特征选取的索引
        self.feature_index = -1


def Maxclass(dataset):
    "计算数据集中最大的类"
    categories = len(type(dataset.label_type()(0)))
    p = [0]*categories

    for d in dataset.sample_list:
        p[d[1].value] +=1

    return dataset.label_type()(p.index(max(p)))


def MaxGainByFeature(dataset, A):
    "计算A中各特征对数据集信息增益最大的一项,返回特征类型和增益值"
    gains = []
    for i in range(len(A)):
        gains.append(gain(dataset, A[i]))

    #print(gains)
    return [A[gains.index(max(gains))], max(gains)]

def ID3(root):
    dataset = root.D
    A = root.A
    el = root.el

    # 数据集属于同一类
    same = is_same_kind(dataset)
    if same[0]==1:
        root.label = same[1]
        root.leaf = 1
        return root

    # 特征集为空
    if len(A)==0:
        root.label = Maxclass(dataset)
        root.leaf = 1
        return root

    # 计算信息增益最大的特征
    max_gain_feature = MaxGainByFeature(dataset, A)
    if max_gain_feature[1]<el:
        root.label = Maxclass(dataset)
        root.leaf = 1
        return root

    #
    MaxFea = max_gain_feature[0]

    MaxFeaIndex = dataset.feature_index(MaxFea)
    root.feature_index = MaxFeaIndex
    categories = len(MaxFea)

    D_i = []
    for i in range(categories):
        D_i.append(DataSet())
    for i in range(dataset.size()):
        s = dataset.getSample(i)
        D_i[  s[0][MaxFeaIndex].value  ].addSample(s)
    # 删除空集
    D_i = [i for i in D_i if i.size() != 0]
    for _d in range(len(D_i)):
        x = D_i[_d].feature_value(0, MaxFea(0))
        root.dict[x] = _d

    tmp_A = [a for a in A if type(a(0)) != type(MaxFea(0))]

    for i in range(len(D_i)):
        tmp_node = Node()
        tmp_node.D = D_i[i]
        tmp_node.A = tmp_A
        tmp_node.el = el
        tmp_node.label= Maxclass(D_i[i])
        root.child.append(tmp_node)

    for r in root.child:
        ID3(r)
    return root

def pred(root, x):
    "预测分类"
    if root.leaf == 1:
        return root.label

    # 当前节点划分的特征
    fea_index = root.feature_index

    # 获取样本中特征的值
    feature_value = x[fea_index]

    # 获取该值被划分到的子节点
    xx =root.dict[feature_value]
    subnode = root.child[root.dict[feature_value]]

    # 递归分类
    return pred(subnode, x)

def tree_traverse(root, level = 0):
    "树的遍历"

    # 输出当前节点的信息
    print(level, "feature index", root.feature_index)
    print(level, "node label",root.label);

    if root.leaf==1:
        return

    # 遍历子节点
    for i in root.child:
        tree_traverse(i, level+1)
    return

class decision_tree:
    "决策树"
    def __init__(self, dataset, A, el):
        self.root = Node()
        self.root.D = dataset
        self.root.A = A
        self.root.el = 0

    def train(self):
        self.root = ID3(self.root)

    def pred(self, x):
        return pred(self.root, x)