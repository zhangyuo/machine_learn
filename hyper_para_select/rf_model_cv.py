#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2021-01-06 14:24
# @Author   : Zhangyu
# @Email    : zhangycqupt@163.com
# @File     : rf_model_cv.py
# @Software : PyCharm
# @Desc     :
"""

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from hyper_para_select.utils import regression_valuation

x = load_boston().data
y = load_boston().target

x = MinMaxScaler().fit_transform(x)

# 不管任何参数，都用默认的，拟合下数据看看
best_params = {
    # "criterion": "mse",
    # "max_depth": 5,
    # "min_samples_split": 100,
    # "min_samples_leaf": 20,
    # "max_features": 'auto',
    # "min_weight_fraction_leaf": 0.1,
    "random_state": 100,
    "n_jobs": 12,
    "verbose": 0,
    # "oob_score": True
}
rf = RandomForestRegressor(**best_params)
rf.fit(x, y)
# print(rf.oob_score_)
regression_valuation(rf, x, y)


def cv_process(best_params, params_grid):
    regressor = RandomForestRegressor(**best_params)
    gsearch = GridSearchCV(estimator=regressor,
                           param_grid=params_grid,
                           scoring='neg_mean_absolute_error',
                           cv=5)
    gsearch.fit(x, y)
    print('最佳参数：', gsearch.best_params_, gsearch.best_score_)

    best_params.update(gsearch.best_params_)
    rf = gsearch.best_estimator_
    # rf.fit(x, y)
    # print(rf.oob_score_)
    regression_valuation(rf, x, y)


print("1、首先对n_estimators进行网格搜索:")
params = {'n_estimators': range(10, 200, 10)}
cv_process(best_params, params)

print("2、对max_depth和min_samples_split进行网格搜索:")
params = {'max_depth': range(3, 9, 1),
          'min_samples_split': range(50, 201, 20)}
cv_process(best_params, params)

print("3、对min_samples_split和min_samples_leaf进行网格搜索:")
params = {'min_samples_split': range(80, 150, 20),
          'min_samples_leaf': range(10, 60, 10)}
cv_process(best_params, params)

print("4、对max_featuresf进行网格搜索:")
params = {'max_features': range(3, 11, 2)}
cv_process(best_params, params)
print(best_params)
