#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2020-12-30 16:21
# @Author   : Zhangyu
# @Email    : zhangycqupt@163.com
# @File     : regression.py
# @Software : PyCharm
# @Desc     :
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# 加载数据
data = load_boston()
# 分割数据
train_x, test_x, train_y, test_y = train_test_split(data.data, data.target, test_size=0.25, random_state=33)
# 使用AdaBoost回归模型
regressor = AdaBoostRegressor()
# model = AdaBoostRegressor(base_estimator=None,
#                           n_estimators=50,
#                           learning_rate=1.0,
#                           loss='linear',
#                           random_state=1)
regressor.fit(train_x, train_y)
pred_y = regressor.predict(test_x)
mse = mean_squared_error(test_y, pred_y)
print("房价预测结果 ", pred_y)
print("均方误差 = ", round(mse, 2))

# 使用决策树回归模型
dec_regressor = DecisionTreeRegressor()
dec_regressor.fit(train_x, train_y)
pred_y = dec_regressor.predict(test_x)
mse = mean_squared_error(test_y, pred_y)
print("决策树均方误差 = ", round(mse, 2))
# 使用KNN回归模型
knn_regressor = KNeighborsRegressor()
knn_regressor.fit(train_x, train_y)
pred_y = knn_regressor.predict(test_x)
mse = mean_squared_error(test_y, pred_y)
print("KNN均方误差 = ", round(mse, 2))
