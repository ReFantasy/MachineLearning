/*
 *    This file is under the MIT license.
 *
 *    Copyright (c) 2020 ReFantasy.cn
 *
 *    Created by tandongliang on 2020/3/15.
 *
 *    Description：
 */

#include <type_traits>
#include "test.h"
#include "./src/tuple_helper.h"
using namespace std;

void Example_4_2()
{
    constexpr int S = 1;
    constexpr int M = 2;
    constexpr int L = 3;

    // 创建数据集
    constexpr int dims = 2;
    constexpr int MaxFeaValue = 3;
    constexpr int classes = 2;

    // 标签重新映射到连续整形 y = -1 -> 0,  y = 1 -> 1
    DataSet<std::array<int, dims>, int> d;

    d.Insert({1, S}, 0);
    d.Insert({1, M}, 0);
    d.Insert({1, M}, 1);
    d.Insert({1, S}, 1);
    d.Insert({1, S}, 0);

    d.Insert({2, S}, 0);
    d.Insert({2, M}, 0);
    d.Insert({2, M}, 1);
    d.Insert({2, L}, 1);
    d.Insert({2, L}, 1);

    d.Insert({3, L}, 1);
    d.Insert({3, M}, 1);
    d.Insert({3, M}, 1);
    d.Insert({3, L}, 1);
    d.Insert({3, L}, 0);

    // 创建分类器并传入数据集
    Bayes<dims, classes, MaxFeaValue> bayes(d, 1);

    // 训练
    bayes.Train();

    // 输出先验概率
    std::cout << "先验概率" << std::endl;
    auto prior = bayes.GetPriorProbability();
    for (int y = 0; y < prior.size(); y++)
    {
        std::cout << "P(Y=" << y << ")=" << prior[y] << std::endl;
    }
    std::cout << std::endl;

    // 输出条件概率
    std::cout << "条件概率(SML分别用123替换)" << std::endl;
    auto condi = bayes.GetConditionProbability();
    auto f = [](int i) {if(i==1)return "S";else if(i==2)return "M";else return "L"; };
    auto g = [](int y) {if(y==0)return -1;else return 1; };
    for (int y = 0; y < prior.size(); y++)
    {
        for (int j = 0; j < dims; j++)
        {
            for (int L = 1; L <= MaxFeaValue; L++)
            {
                if (j == 1)
                {
                    std::cout << "P(X"
                              << "^" << j + 1 << "=" << f(L) << "|Y=" << g(y) << ") = " << condi[y][j][L] << std::endl;
                }
                else
                {
                    std::cout << "P(X"
                              << "^" << j + 1 << "=" << L << "|Y=" << g(y) << ") = " << condi[y][j][L] << std::endl;
                }
            }
        }
    }

    // 预测
    std::cout << "预测" << std::endl;
    std::array<int, 2> x{2, S};
    int Y = bayes.Pred(x);
    std::cout << "P(x=(2,S)) = ";
    std::cout << g(Y) << std::endl;
}



void test()
{
	// 定义数据特征
	enum class Age:int{Young=0, Middle, Old};
	enum class Work:int{Yes=0, No};
	enum class House:int{Yes=0, No};
	enum class Credit:int{Nice=0, Good, Ordinary};

	enum class Category:int{Yes=0, No};

	using Data = std::tuple< Feature<3,Age>,Feature<2,Work>,Feature<2,House>,Feature<3,Credit> > ;

	// 特征集
	struct FeatureSet
	{
		Data feature;
		std::vector<bool> used = std::vector<bool>(4, false);  // 初始四个特征都没有用于分类
	};

	// 数据集
	DataSet<Data, Feature<2,Category>> dataset;

	dataset.Insert(Data(Age::Young, Work::No, House::No, Credit::Ordinary),Feature<2,Category>(Category::No));
	dataset.Insert(Data(Age::Young, Work::No, House::No, Credit::Good),Feature<2,Category>(Category::No));
	dataset.Insert(Data(Age::Young, Work::Yes, House::No, Credit::Good),Feature<2,Category>(Category::Yes));
	dataset.Insert(Data(Age::Young, Work::Yes, House::Yes, Credit::Ordinary),Feature<2,Category>(Category::Yes));
	dataset.Insert(Data(Age::Young, Work::No, House::No, Credit::Ordinary),Feature<2,Category>(Category::No));

	dataset.Insert(Data(Age::Middle, Work::No, House::No, Credit::Ordinary),Feature<2,Category>(Category::No));
	dataset.Insert(Data(Age::Middle, Work::No, House::No, Credit::Good),Feature<2,Category>(Category::No));
	dataset.Insert(Data(Age::Middle, Work::Yes, House::Yes, Credit::Good),Feature<2,Category>(Category::Yes));
	dataset.Insert(Data(Age::Middle, Work::No, House::Yes, Credit::Nice),Feature<2,Category>(Category::Yes));
	dataset.Insert(Data(Age::Middle, Work::No, House::Yes, Credit::Nice),Feature<2,Category>(Category::Yes));

	dataset.Insert(Data(Age::Old, Work::No, House::Yes, Credit::Nice),Feature<2,Category>(Category::Yes));
	dataset.Insert(Data(Age::Old, Work::No, House::Yes, Credit::Good),Feature<2,Category>(Category::Yes));
	dataset.Insert(Data(Age::Old, Work::Yes, House::No, Credit::Good),Feature<2,Category>(Category::Yes));
	dataset.Insert(Data(Age::Old, Work::Yes, House::No, Credit::Nice),Feature<2,Category>(Category::Yes));
	dataset.Insert(Data(Age::Old, Work::No, House::No, Credit::Ordinary),Feature<2,Category>(Category::No));




	std::cout<<Entropy(dataset)<<std::endl;

	std::cout<<ConditionEntropy(dataset, Feature<3,Credit>())<<std::endl;



}
