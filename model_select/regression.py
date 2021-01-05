#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2020-12-30 13:58
# @Author   : Zhangyu
# @Email    : zhangycqupt@163.com
# @File     : regression.py
# @Software : PyCharm
# @Desc     :
"""
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from hyper_para_select.lgb_model_cv import data_prepare
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def model_selection():
    """
    https://cloud.tencent.com/developer/article/1609448
    :return:
    """
    [X_train, y_train] = data_prepare()[0:2]
    num_folds = 10
    seed = 7

    # 把所有模型写到一个字典中
    models = {}
    models['Linear_regression'] = LinearRegression()
    models['Ridge'] = Ridge()
    models['LASSO'] = Lasso()
    models['DecisionTree'] = DecisionTreeRegressor()
    models['RandomForest'] = RandomForestRegressor()
    models['GradientBoosting'] = GradientBoostingRegressor()
    models['XGB'] = XGBRegressor(n_estimators=100, objective='reg:squarederror')
    models['LGB'] = LGBMRegressor(n_estimators=100)
    # models['SVR'] = SVR()   # 支持向量机运行不出来

    results = []
    for key in models:
        kfold = KFold(n_splits=num_folds, random_state=seed)
        cv_result = cross_val_score(models[key], X_train, y_train, cv=kfold, scoring=make_scorer(mean_absolute_error))
        results.append(cv_result)
        print('%s: %f (%f)' % (key, cv_result.mean(), cv_result.std()))

    # 评估算法 --- 箱线图
    fig1 = plt.figure(figsize=(15, 10))
    fig1.suptitle('Algorithm Comparison')
    ax = fig1.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(models.keys())
    plt.show()


if __name__ == "__main__":
    model_selection()
