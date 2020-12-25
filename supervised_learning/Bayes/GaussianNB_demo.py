#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2018/9/30 10:13
@Author  : Zhangyu
@Email   : zhangycqupt@163.com
@File    : GaussianNB_demo.py
@Software: PyCharm
@Github  : zhangyuo
"""

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

"""
data1 鸢尾花数据
"""
iris = datasets.load_iris()  # 导入数据集
x = iris.data  # 获得其特征向量
y = iris.target  # 获得样本label
# print(x, '\n', y, '\n')

"""
one hot 编码数据
"""
# x = [['重庆' '美丽' '漂亮']['美丽' '重庆' '漂亮']['北京' '快速' '发展']['发展' '快速' '北京']]
x = [[1, 2, 3], [2, 1, 3], [4, 5, 6], [6, 5, 4]]
y = [1, 1, 2, 2]

gnb = GaussianNB()
y_pred = gnb.fit(x, y).predict(x)
print('Number of mislabeled points out of a total %d points : %d'
      % (len(x), (y != y_pred).sum()))
