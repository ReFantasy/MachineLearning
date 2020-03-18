# -*- coding: utf-8 -*-
#
# 定义数据类型和数据集类
#
#


from enum import Enum, IntEnum

# 定义特征
class Age(Enum):
    Young = 0
    Middle = 1
    Old = 2

class Work(Enum):
    Yes = 1
    No = 0

class House(Enum):
    Yes = 1
    No = 0

class Credit(Enum):
    Ordinary = 0
    Good = 1
    Nice = 2

class Category(Enum):
    Yes = 1
    No = 0


class DataSet():
    "数据集样本列表组成，每条样本又由数据和标签组成的二元列表构成，样本的数据部分由特征列表组成"

    def __init__(self):
        self.sample_list = []

    def getData(self, n):
        """返回数据集第n个样本的数据部分"""
        return self.sample_list[n][0]

    def getLabel(self, n):
        """返回数据集第n个样本的标签部分"""
        return self.sample_list[n][1]

    def getSample(self, n):
        """返回数据集第n个样本"""
        return self.sample_list[n]

    def addSample(self,s):
        self.sample_list.append(s)

    def print(self):
        print(self.sample_list)

    def size(self):
        return len(self.sample_list)

    def feature_type(self, n):
        "第n维特征的数据类型"
        return type(self.sample_list[0][0][n])
    def label_type(self):
        "返回标签数据类型"
        return type(self.sample_list[0][1])


    def feature_value(self, n, feature_type):
        ""
        #print(type(feature_type))
        # 计算该特征所属维度
        fea_n = -1
        fea_len = len(self.getData(0))
        for i in range(fea_len):
            if type(feature_type) == type(self.getData(0)[i]):
                fea_n = i
                break
        return self.getData(n)[fea_n]

    def feature_index(self, feature_type):
        fea_n = -1
        fea_len = len(self.getData(0))
        for i in range(fea_len):
            if type(feature_type(0)) == type(self.getData(0)[i]):
                fea_n = i
                break
        return fea_n

    def print(self):
        for i in range(len(self.sample_list)):
            print(self.sample_list[i])