/*
 *    This file is under the MIT license.
 *
 *    Copyright (c) 2020 ReFantasy.cn
 *
 *    Created by tandongliang on 2020/3/16.
 *
 *    Description：
 */

#ifndef DECISION_TREE_H
#define DECISION_TREE_H
#include <iostream>
#include <vector>
#include <string>
#include "dataset.hpp"
#include "tuple_helper.h"

/**
 * 特征类型 基于枚举类型并携带枚举类型相关信息
 * @tparam ENUMS 特征可取值的个数
 * @tparam T 基本枚举类型
 */
template <int ENUMS, typename T>
class Feature
{
public:
	// 构造以及类型转换
	Feature()= default;
	Feature(const T value):_enum_value(value){}
	Feature(const int n):_enum_value(T(n)){}

	explicit operator int()const
	{
		return static_cast<int>(_enum_value);
	}

	// 重载等于运算符
	bool operator==(const Feature &rhs)const
	{
		return (_enum_value==rhs._enum_value);
	}


	constexpr static int enums = ENUMS;
private:
	T _enum_value;
};


/**
 * 熵函数，其值越大，表示概率为p的随机变量X的不确定性就越大
 * @param p 概率分布向量
 * @return 熵的大小
 */
double Entropy(const std::vector<double> &p);


/**
 * 计算数据集的经验熵 H(d)
 * @tparam D 数据集数据类型
 * @tparam L 数据集标签类型
 * @param d 数据集实例
 * @return 数据集在标签上的经验熵
 */
template <typename D, typename L>
double EmpiricalEntropy(const DataSet<D, L> &d)
{
	// 概率分布向量
	std::vector<double> p;

	int enum_range = L::enums;

	// 计算每个类别下的个数
	std::vector<int> Ck(enum_range,0);
	for(int i = 0;i<d.size();i++)
	{
		Ck[static_cast<int>(d[i].second)]++;
	}
	// 计算每个类别的概率
	for(int j=0; j<enum_range; j++)
	{
		p.push_back(((double)Ck[j])/d.size());
	}

	return Entropy(p);

}

/**
 * 计算数据集在给定给定特征下的条件熵
 * @tparam D 数据集每个元素数据部分的数据类型
 * @tparam L 数据集每个元素标签部分的数据类型
 * @tparam FEA_TYPE 特征类型
 * @param d 数据集实例
 * @return 特征FEA_TYPE下的条件熵
 */
template <typename D, typename L, typename FEA_TYPE>
double ConditionEntropy(const DataSet<D,L> &d, FEA_TYPE)
{
	// 特征可取值个数
	int enum_range = FEA_TYPE::enums;

	std::vector<double> p(enum_range,0);
	std::vector<double> H(enum_range,0);

	// 遍历每个特征
	for(int feature_value = 0; feature_value<enum_range; feature_value++)
	{
		int count = 0;
		DataSet<D,L> tmp_dataset;

		// 遍历数据集 寻找符合该特征的样本
		for(int i = 0;i<d.size();i++)
		{
			if(FindElementIndexInTuple(d[i].first, FEA_TYPE(feature_value))>=0)
			{
				tmp_dataset.Insert(d[i].first, d[i].second);
				count++;
			}
		}


		H[feature_value] = EmpiricalEntropy(tmp_dataset);
		p[feature_value] = ((double)count)/d.size();
	}

	double result = 0;
	for(int i=0; i<enum_range; i++)
	{
		result +=(p[i]*H[i]);
	}

	return result;
}

/**
 * 判断数据集D中的数据是否属于同一类
 * @tparam D 数据集每个元素数据部分的数据类型
 * @tparam L 数据集每个元素标签部分的数据类型
 * @param d 数据集实例
 * @param [out] category 如果同一类，输出该类别
 * @return
 */
template <typename D, typename L>
bool IsSameKind(const DataSet<D, L> d, L &category)
{
	if(d.size()<=0)
		return false;
	if(d.size()==1)
	{
		category = d[0].second;
		return true;
	}

	bool is_same = true;
	category = d[0].second;
	for(int i = 1; i<d.size(); i++)
	{
		if(d[i].second != d[i-1].second)
		{
			is_same = false;
			break;
		}
	}
	return is_same;
}


/**
 * 计算数据集在给定给定特征下的条件熵
 * @tparam D 数据集每个元素数据部分的数据类型
 * @tparam L 数据集每个元素标签部分的数据类型
 * @tparam FEA_TYPE 特征类型
 * @param d 数据集实例
 * @return 特征FEA_TYPE下的条件熵
 */
template <typename D, typename L, typename FEA_TYPE>
double GainInfo(const DataSet<D,L> &d, FEA_TYPE)
{
	return EmpiricalEntropy(d)-ConditionEntropy(d, FEA_TYPE{});
}

/**
 * 特征集
 * @tparam D 数据集每个元素数据部分的数据类型（即特征）
 * @tparam N 特征集中包含的特征个数
 */
template <typename D,int N>
class A
{
public:
	A()
	{
		is_deleted = std::vector<bool>(N, false);
	}

	/**
	 * 标记特征为已经删除
	 * @param n 删除索引
	 */
	void Delete(int n)
	{
		is_deleted[n] = true;
	}
	size_t size()const
	{
		return N;
	}

	bool IsDeleted(int n)const
	{
		return is_deleted[n];
	}
	D GetThis()__const
	{
		return d;
	}
private:
	D d;
	std::vector<bool> is_deleted;
	static constexpr int dims = N;
};

template <typename D, typename L, int N>
double MaxGain(const DataSet<D,L> &d, A<D, N> _A)
{
	int max_index = 0;
	double cur_gain = 0;

	for(int i = 0;i<_A.size();i++)
	{
		if(!_A.IsDeleted(i))
		{

		}
	}


}
#endif //DECISION_TREE_H
