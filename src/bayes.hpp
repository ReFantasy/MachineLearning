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

/**
 * 贝叶斯分类器
 * @tparam DIMS 样本特征的维度
 * @tparam CLASSES_N 分类的类别:0,1,2,...,ClassesN-1
 * @tparam MAXFEA_VALUE 特征值得最大取值
 * @tparam FEATURE_TYPE 特征数据类型
 */
template <int DIMS, int CLASSES_N, int MAXFEA_VALUE, typename FEATURE_TYPE = int>
class Bayes
{
public:
    Bayes() = default;

    /**
     *
     * @param d 数据集，由若干条记录组成，每条记录由特征向量和标签组成
     * @param lambda 贝叶斯估计参数，默认为0，通常取1
     * @param Sj Sj[i]表示第i维特征可能取到的值得个数
     */
    explicit Bayes(const DataSet<std::array<FEATURE_TYPE, DIMS>, int> &d,
          double lambda = 0,
          const std::vector<int> &Sj = std::vector<int>(DIMS, MAXFEA_VALUE)) : _dataset(d), _lambda(lambda), _Sj(Sj)
    {
        int MaxK = d.GetRecord(0).second;
        for (int i = 0; i < d.size(); i++)
        {
            if (d.GetRecord(i).second > MaxK)
                MaxK = d.GetRecord(i).second;
        }
        K = MaxK;
    }

    /**
     * 训练分类器
     */
    void Train()
    {
        ComputePrior();
        ComputeConditionPrior();
    }

    /**
     * 预测贝叶斯分类
     * @param x 预测样本的特征向量
     * @return 分类结果
     */
    int Pred(const std::array<FEATURE_TYPE, DIMS> &x)
    {
        double result[CLASSES_N] = {0};
        for (int k = 0; k < CLASSES_N; k++)
        {
            double prob = _prior_probability[k];
            for (int j = 0; j < DIMS; j++)
            {
                prob *= _condition_probability[k][j][x[j]];
            }

            result[k] = prob;
            //std::cout<<prob<<std::endl;
        }

        int max_index = 0;
        for (int k = 0; k < CLASSES_N; k++)
        {
            if (result[k] > result[max_index])
                max_index = k;
        }

        return max_index;
    }

    /**
    *
    * @return 先验概率计算结果
    */
    std::array<double, CLASSES_N> GetPriorProbability()const{return _prior_probability;}

    /**
     *
     * @return 条件概率计算结果
     */
    std::array<std::array<std::array<double, MAXFEA_VALUE + 1>, DIMS>, CLASSES_N> GetConditionProbability()const
    {return _condition_probability;}

private:
    /**
     * 计算先验概率
     */
    void ComputePrior()
    {
        for (int k = 0; k < CLASSES_N; k++)
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

    /**
     * 计算条件概率
     */
    void ComputeConditionPrior()
    {
        for (int k = 0; k < CLASSES_N; k++)
        {
            for (int j = 0; j < DIMS; j++)
            {
                for (int L = 0; L <= MAXFEA_VALUE; L++)
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
    DataSet<std::array<FEATURE_TYPE, DIMS>, int> _dataset;

    // 先验概率
    std::array<double, CLASSES_N> _prior_probability = {0};

    // 条件概率
    std::array<std::array<std::array<double, MAXFEA_VALUE + 1>, DIMS>, CLASSES_N> _condition_probability = {0};

    double _lambda = 0;

    // 特征值可取值得个数
    std::vector<int> _Sj = std::vector<int>(DIMS, MAXFEA_VALUE);

    // 样本标签可取到的值的个数
    int K = 0;
};


#endif //__BAYES_H__
