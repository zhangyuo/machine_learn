#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/3/22 15:52
@Author  : Zhangyu
@Email   : zhangycqupt@163.com
@File    : data_load.py
@Software: PyCharm
@Github  : zhangyuo
"""

from sklearn import datasets
"""
load_boston([return_X_y])	Load and return the boston house-prices dataset (regression).
load_iris([return_X_y])	Load and return the iris dataset (classification).
load_diabetes([return_X_y])	Load and return the diabetes dataset (regression).
load_digits([n_class, return_X_y])	Load and return the digits dataset (classification).
load_linnerud([return_X_y])	Load and return the linnerud dataset (multivariate regression).
load_wine([return_X_y])	Load and return the wine dataset (classification).
load_breast_cancer([return_X_y])	Load and return the breast cancer wisconsin dataset (classification).
"""

# 鸢尾花数据集
iris = datasets.load_iris()
#  shape (n_samples, n_features)
x = iris.data
y = iris.target
print(x)
print(y)

# 字符数字
digits = datasets.load_digits()
x = digits.data
y = digits.target
print(x)
print(y)
digits.images[0]
