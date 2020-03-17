/*
 *    This file is under the MIT license.
 *
 *    Copyright (c) 2020 ReFantasy.cn
 *
 *    Created by tandongliang on 2020/3/17.
 *
 *    Description：
 */

#ifndef TUPLE_HELPER_H
#define TUPLE_HELPER_H
#include <iostream>


/**
 *    Tuple类辅助函数
 */

template <class Tuple, std::size_t N>
struct TuplePrinter
{
	static  void print(const Tuple &t)
	{
		TuplePrinter<Tuple, N-1>::print(t);
		std::cout<<", "<<std::get<N-1>(t);
	}
};

template <class Tuple>
struct TuplePrinter<Tuple,1>
{
	static void print(const Tuple &t)
	{
		std::cout<<std::get<0>(t);
	}
};

template <typename ...Args>
void PrintTuple(const std::tuple<Args...> &t)
{
	std::cout<<"(";
	TuplePrinter<decltype(t), sizeof...(Args)>::print(t);
	std::cout<<")\n";
}


/**
 * 根据元素值获取索引位置
 */
// 对于可转换类型则直接进行比较
template <typename T, typename U>
typename std::enable_if<std::is_convertible<T,U>::value||std::is_convertible<U,T>::value, bool>::type
Compare(const T &t, const U &u)
{
	return t==u;
}

// 不能转换的直接返回false
bool Compare(...);

template <int I, typename T, typename ...Args>
struct find_index
{
	static int call(std::tuple<Args...> const &t, T &&val)
	{
		return (Compare(std::get<I-1>(t),val)?I-1:
		find_index<I-1,T, Args...>::call(t,std::forward<T>(val)));
	}
};

template <typename T, typename ...Args>
struct find_index<0,T,Args...>
{
	// 递归终止，如果找到则返回0， 否则返回-1
	static int call(std::tuple<Args...> const &t, T&&val)
	{
		return Compare(std::get<0>(t),val)?0:-1;
	}
};

// 辅助函数 简化调用
template <typename T, typename ...Args>
int FindIndex(std::tuple<Args...> const &t, T&&val)
{
	return find_index<sizeof...(Args), T, Args...>::call(t, std::forward<T>(val));
}

/**
 * 在运行期根据索引获取索引位置的元素
 */
template <size_t k, typename  Tuple>
typename std::enable_if< k==std::tuple_size<Tuple>::value >::type
GetArgByIndex(size_t index,  const Tuple &tp)
{
	throw std::invalid_argument("arg index out of range");
}

template <size_t k=0, typename  Tuple>
typename std::enable_if<(k<std::tuple_size<Tuple>::value)>::type
GetArgByIndex(size_t index,  const Tuple &tp)
{
	if(k==index)
	{
		std::cout<<std::get<k>(tp)<<std::endl;
	}
	else
	{
		GetArgByIndex<k+1>(index, tp);
	}
}

#endif //TUPLE_HELPER_H
