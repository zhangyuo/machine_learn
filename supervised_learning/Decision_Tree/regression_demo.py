#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2020-12-30 16:41
# @Author   : Zhangyu
# @Email    : zhangycqupt@163.com
# @File     : regression_demo.py
# @Software : PyCharm
# @Desc     :
"""

from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# 加载数据
data = load_boston()
# 分割数据
train_x, test_x, train_y, test_y = train_test_split(data.data, data.target, test_size=0.25, random_state=33)
# 使用决策树回归模型
dec_regressor = DecisionTreeRegressor()
dec_regressor.fit(train_x, train_y)
pred_y = dec_regressor.predict(test_x)
mse = mean_squared_error(test_y, pred_y)
print("决策树均方误差 = ", round(mse, 2))
