#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019-04-04 16:39
@Author  : zhangyu
@Contact : zhangycqupt@163.com
@File    : nearest_neighbors.py
@Software: PyCharm
@Site    : https://github.com/zhangyuo
"""

"""
无监督最近邻方法是很多学习方法的基础，特别是流形学习和谱聚类。
https://www.jianshu.com/p/778d39132d66

NearestNeighbors 执行无监督的最近邻方法，有三种不同的最近邻算法：BallTree、KDTree、a brute-force algorithm based on routines in sklearn.metrics.pairwise，
邻居的搜索算法通过关键词 ‘algorithm’ 控制，选项包括['auto', 'ball_tree', 'kd_tree', 'brute']，当设置为‘auto’时，算法将通过训练数据决定最好的方法。
Warning：在最近邻算法中，当有两个点和预测点的距离相同但标签不同时，结果将依赖点在训练数据中的顺序。
"""
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt


def n_neighbors():
    """
    找到两组数据集中最近邻点的简单任务
    :return:
    """
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    # n_neighbors 指定包括本样本在内距离本样本最近的 n 个点
    # algorithm   指定最临近算法
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    # representing the cosine distances to each point
    # len(X)*n_neighbors的向量，每一行表示最邻近的n_neighbors个样本距离本样本点的距离
    print(distances)
    # Indices of the approximate nearest points in the population matrix
    # len(X)*n_neighbors的向量，每一行表示距离本样本距离由小到大的样本的index
    print(indices)

    # k个最近点的下标，按升序排列
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    plt.title("Unsupervised nearest neighbors")
    plt.show()


if __name__ == '__main__':
    n_neighbors()
