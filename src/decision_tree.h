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
 * 特征数据类型 由基本枚举类型构成
 * @tparam ENUMS 枚举类型的取值个数
 * @tparam T 特征的枚举类型
 */
template <int ENUMS, typename T>
class Feature
{
public:
	using value_type = T;
	constexpr static int enums = ENUMS;
	Feature()= default;
	Feature(T value):_value(value){}

	// 类型转换
	Feature(int n):_value(T(n)){}

	explicit operator int()const
	{
		return static_cast<int>(_value);
	}

	bool operator==(const Feature &rhs)const
	{
		return (_value==rhs._value);
	}

	std::ostream& operator<<(const Feature &f)
	{
		std::cout<< static_cast<int>(f._value);
		return std::cout;
	}

	static const size_t id ;
private:
	T _value;
};

template <int ENUMS, typename T>
const size_t Feature<ENUMS,T>::id = std::hash<std::string>()(std::string(typeid(Feature<ENUMS,T>).name()));


/**
 * 熵函数，其值越大，表示概率为p的随机变量X的不确定性就越大
 * @param p 概率分布向量
 * @return 概率为p条件下的熵的大小
 */
double Entropy(const std::vector<double> &p);

//
/**
 * 计算数据集的经验熵 H(d)
 * @tparam D 数据集数据类型
 * @tparam L 数据集标签类型
 * @param d 数据集实例
 * @return 数据集在标签上的熵
 */
template <typename D, typename L>
double Entropy(const DataSet<D,L> &d)
{
	std::vector<double> p;
	int cn = L::enums;
	std::vector<int> Ck(cn,0);
	for(int i = 0;i<d.size();i++)
	{
		Ck[static_cast<int>(d[i].second)]++;
	}
	for(int j=0;j<cn;j++)
	{
		p.push_back(((double)Ck[j])/d.size());
	}

	return Entropy(p);

}
/**
 * 计算数据集在给定给定特征下的条件熵
 * @tparam D 数据集数据类型
 * @tparam L 数据集标签类型
 * @tparam FEA_TYPE 特征类型
 * @param d 数据集实例
 * @return 条件熵
 */
template <typename D, typename L, typename FEA_TYPE>
typename std::enable_if<std::is_enum<typename L::value_type>::value,double>::type
ConditionEntropy(const DataSet<D,L> &d, FEA_TYPE)
{
	// 特征可取值个数
	int n = FEA_TYPE::enums;
	std::vector<double> p(n,0);
	std::vector<double> H(n,0);

	// 遍历特征的每个值
	for(int c = 0;c<n; c++)
	{
		int count = 0;
		DataSet<D,L> tmp_dataset;
		for(int i = 0;i<d.size();i++)
		{

			if(FindIndex(d[i].first, FEA_TYPE(c))>=0)
			{
				tmp_dataset.Insert(d[i].first, d[i].second);
				count++;
			}
		}
		H[c] = Entropy(tmp_dataset);
		p[c] = ((double)count)/d.size();
	}

	double result = 0;
	for(int i=0;i<n;i++)
	{
		result +=(p[i]*H[i]);
	}
	return result;
}


// 判断数据集D中的数据是否属于同一类
template <typename D, typename L>
bool IsSameClass(const DataSet<D,L> d, L &category)
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
		}
	}
	return is_same;
}

#endif //DECISION_TREE_H
