/*
 *
 *    朴素贝叶斯分类器 李航《统计学习方法》代码实现
 *
 *    该类仅用于研究贝叶斯分类算法，样本特征的不同特征默认采用同种数据类型（整型）表示
 *    样本类别标签必须是从零开始的连续整数
 *
 *    License: MIT
 * 
 *    ©TDL 2020.03.13
 *    Contact:refantasy.cn
 * 
 */

#ifndef __BAYES_H__
#define __BAYES_H__
#include <vector>
#include <array>
#include "dataset.hpp"

/*
 * *\brief 贝叶斯分类器
 *  \param dims 样本特征的维度
 *  \param ClassesN 分类的类别:0,1,2,...,ClassesN-1
 *  \param MaxFeaValue 特征值得最大取值
 */
template <int dims, int ClassesN, int MaxFeaValue, typename FEATURE_TYPE = int>
class Bayes
{
    friend void Example_4_2();

public:
    Bayes() = default;

    /* *\param d 数据集，由若干条记录组成，每条记录由特征向量和标签组成
     *  \param lambd 贝叶斯估计参数，默认为0，通常取1
     *  \param Sj Sj[i]表示第i维特征可能取到的值得个数
     */
    Bayes(const DataSet<std::array<FEATURE_TYPE, dims>, int> &d,
          double lambda = 0,
          std::vector<int> Sj = std::vector<int>(dims, MaxFeaValue)) : _dataset(d), _lambda(lambda), _Sj(Sj)
    {
        int MaxK = d.GetRecord(0).second;
        for (int i = 0; i < d.size(); i++)
        {
            if (d.GetRecord(i).second > MaxK)
                MaxK = d.GetRecord(i).second;
        }
        K = MaxK;
    }

    // 训练
    void Train()
    {
        ComputePrior();
        ComputeConditionPrior();
    }

    // 预测
    int Pred(const std::array<FEATURE_TYPE, dims> &x)
    {
        double result[ClassesN] = {0};
        for (int k = 0; k < ClassesN; k++)
        {
            double prob = _prior_probability[k];
            for (int j = 0; j < dims; j++)
            {
                prob *= _condition_probability[k][j][x[j]];
            }

            result[k] = prob;
            //std::cout<<prob<<std::endl;
        }

        int max_index = 0;
        for (int k = 0; k < ClassesN; k++)
        {
            if (result[k] > result[max_index])
                max_index = k;
        }

        return max_index;
    }

private:
    // 计算先验概率
    void ComputePrior()
    {
        for (int k = 0; k < ClassesN; k++)
        {
            int I = 0;
            for (int i = 0; i < _dataset.size(); i++)
            {
                if (_dataset.GetRecord(i).second == k)
                    I++;
            }
            _prior_probability[k] = ((double)I + _lambda) / (_dataset.size() + (K + 1) * _lambda);
        }
    }

    // 计算条件概率
    void ComputeConditionPrior()
    {
        for (int k = 0; k < ClassesN; k++)
        {
            for (int j = 0; j < dims; j++)
            {
                for (int L = 0; L <= MaxFeaValue; L++)
                {
                    int p_x_y = 0;
                    int y_c_k = 0;
                    for (int i = 0; i < _dataset.size(); i++)
                    {
                        if (_dataset.GetRecord(i).second == k)
                        {
                            y_c_k++;
                            if (_dataset.GetRecord(i).first[j] == L)
                                p_x_y++;
                        }
                    }
                    _condition_probability[k][j][L] = ((double)p_x_y + _lambda) / (y_c_k + _Sj[j] * _lambda);
                }
            }
        }
    }

private:
    // 数据集
    DataSet<std::array<FEATURE_TYPE, dims>, int> _dataset;

    // 先验概率
    std::array<double, ClassesN> _prior_probability = {0};

    // 条件概率
    std::array<std::array<std::array<double, MaxFeaValue + 1>, dims>, ClassesN> _condition_probability = {0};

    double _lambda = 0;

    // 特征值可取值得个数
    std::vector<int> _Sj = std::vector<int>(dims, MaxFeaValue);

    // 样本标签可取到的值的个数
    int K = 0;
};

/*
 *
 * 
 * 
 *         测试函数，李航《统计学习方法》例 4-2（Include 4-1）
 * 
 * 
 * 
 */
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
    auto &prior = bayes._prior_probability;
    for (int y = 0; y < prior.size(); y++)
    {
        std::cout << "P(Y=" << y << ")=" << prior[y] << std::endl;
    }
    std::cout << std::endl;

    // 输出条件概率
    std::cout << "条件概率(SML分别用123替换)" << std::endl;
    auto &condi = bayes._condition_probability;
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

#endif //__BAYES_H__
