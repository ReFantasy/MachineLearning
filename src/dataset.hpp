/*
 *
 * 
 *    通用数据集类
 * 
 *    模板参数 D 为样本数据部分的数据类型
 *    模板参数 L 为样本标签部分的数据类型
 *    一个样布数据和一个样本标签组成一条记录
 *    默认使用 vector 容器保存数据记录
 * 
 *    License: MIT
 * 
 *    ©TDL 2020.03.13
 *    Contact:refantasy.cn
 * 
 */

#ifndef __DATASET_HPP__
#define __DATASET_HPP__
#include <utility>
#include <vector>

template <typename D, typename L>
class DataSet
{
public:
    using value_type = D;
    using label_type = L;
    using record_type = std::pair<D, L>;

    DataSet() = default;

    /**
     * 插入一个样本
     * @param d 样本特征向量
     * @param la 样本标签值
     */
    void Insert(const value_type &d, const label_type &la)
    {
        data.push_back({d, la});
    }

    /**
     * 查找一个样本
     * @param n 样本索引
     * @return 一条记录引用
     */
    const record_type& GetRecord(size_t n) const
    {
        return data[n];
    }

    /**
     *
     * @return 数据集大小
     */
    size_t size() const { return data.size(); }

private:
    // 样本集合
    std::vector<record_type> data;
};

#endif //__DATASET_HPP__
