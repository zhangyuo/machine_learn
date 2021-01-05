#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2020-12-30 16:40
# @Author   : Zhangyu
# @Email    : zhangycqupt@163.com
# @File     : regression_demo.py
# @Software : PyCharm
# @Desc     :
"""
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# 加载数据
data = load_boston()
# 分割数据
train_x, test_x, train_y, test_y = train_test_split(data.data, data.target, test_size=0.25, random_state=33)

# 使用KNN回归模型
knn_regressor = KNeighborsRegressor()
knn_regressor.fit(train_x, train_y)
pred_y = knn_regressor.predict(test_x)
mse = mean_squared_error(test_y, pred_y)
print("KNN均方误差 = ", round(mse, 2))
