from ID3 import *

#Work.value
# 创建数据集
dataset = DataSet()

dataset.addSample([[Age.Young, Work.No, House.No, Credit.Ordinary], Category.No])
dataset.addSample([[ Age.Young, Work.No, House.No, Credit.Good], Category.No ])
dataset.addSample([[ Age.Young, Work.Yes, House.No, Credit.Good], Category.Yes ])
dataset.addSample([[ Age.Young, Work.Yes, House.Yes, Credit.Ordinary], Category.Yes ])
dataset.addSample([[ Age.Young, Work.No, House.No, Credit.Ordinary], Category.No ])

dataset.addSample([[ Age.Middle, Work.No, House.No, Credit.Ordinary], Category.No ])
dataset.addSample([[ Age.Middle, Work.No, House.No, Credit.Good], Category.No ])
dataset.addSample([[ Age.Middle, Work.Yes, House.Yes, Credit.Good], Category.Yes ])
dataset.addSample([[ Age.Middle, Work.No, House.Yes, Credit.Nice], Category.Yes ])
dataset.addSample([[ Age.Middle, Work.No, House.Yes, Credit.Nice], Category.Yes ])

dataset.addSample([[ Age.Old, Work.No, House.Yes, Credit.Nice], Category.Yes ])
dataset.addSample([[ Age.Old, Work.No, House.Yes, Credit.Good], Category.Yes ])
dataset.addSample([[ Age.Old, Work.Yes, House.No, Credit.Good], Category.Yes ])
dataset.addSample([[ Age.Old, Work.Yes, House.No, Credit.Nice], Category.Yes ])
dataset.addSample([[ Age.Old, Work.No, House.No, Credit.Ordinary], Category.No ])

# 特征集
A = [Age, Work, House, Credit]

# 决策树
dtree = decision_tree(dataset, A, 0)

# 训练
dtree.train()

# 预测
x = [Age.Middle, Work.Yes, House.No, Credit.Good]
category =  dtree.pred(x)
print("样本预测类别",category)









