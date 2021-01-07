#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2021-01-06 16:49
# @Author   : Zhangyu
# @Email    : zhangycqupt@163.com
# @File     : utils.py
# @Software : PyCharm
# @Desc     :
"""

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from scipy.stats import pearsonr


def regression_valuation(regressor, x, y):
    """
    回归问题评测方法
    :return:
    """
    y_predprob = regressor.predict(x)
    mse = mean_squared_error(y, y_predprob)
    mae = mean_absolute_error(y, y_predprob)
    r2_score1 = r2_score(y, y_predprob)  # 判定系数，其含义是也是解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量的方差变化，值越小则说明效果越差。
    p = pearsonr(y, y_predprob)[0]
    # auc = roc_auc_score(y, y_predprob) # 分类时用
    auc = explained_variance_score(y, y_predprob)  # 解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量的方差变化，值越小则说明效果越差。
    print('mse:', mse, 'mae:', mae, 'r2_score:', r2_score1, '皮尔逊相关系数:', p, 'auc:', auc)
