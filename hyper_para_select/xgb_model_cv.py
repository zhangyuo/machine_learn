#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2021-01-07 10:19
# @Author   : Zhangyu
# @Email    : zhangycqupt@163.com
# @File     : xgb_model_cv.py
# @Software : PyCharm
# @Desc     :
"""
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_boston

from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor

from hyper_para_select.utils import regression_valuation

x = load_boston().data
y = load_boston().target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=100)
xgb_train = xgb.DMatrix(x_train, y_train)
xgb_test = xgb.DMatrix(x_test, y_test)

# 不管任何参数，都用默认的，拟合下数据看看
min_mae = float(10)
best_params = {
    "random_state": 100,
    "n_jobs": -1,
    "verbosity": 1,
    "learning_rate": 0.1
}
regressor = XGBRegressor(**best_params)
regressor.fit(x, y)
regression_valuation(regressor, x, y)


def cv_process(best_params, params_grid):
    regressor = XGBRegressor(**best_params)
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


print('调参n_estimators')
n_estimators = 200
cv_results = xgb.cv(params=best_params,
                    dtrain=xgb_train,
                    num_boost_round=n_estimators,
                    nfold=5,
                    metrics=['rmse', 'mae'],
                    early_stopping_rounds=10,
                    verbose_eval=0,
                    seed=1,
                    stratified=False)
n_estimators = len(cv_results['test-mae-mean'])
min_mae = list(cv_results['test-mae-mean'])[-1]
print('best n_estimators:', n_estimators)
print('best cv score', min_mae)

print('调参max_depth')
for max_depth in range(3, 9, 1):
    params = best_params.copy()
    params['max_depth'] = max_depth
    cv_results = xgb.cv(params=params,
                        dtrain=xgb_train,
                        num_boost_round=n_estimators,
                        nfold=5,
                        metrics=['rmse', 'mae'],
                        early_stopping_rounds=10,
                        verbose_eval=0,
                        seed=1,
                        stratified=False)
    mean_mae = cv_results['test-mae-mean'].min()
    if mean_mae <= min_mae:
        min_mae = mean_mae
        best_params['max_depth'] = max_depth
print(best_params)
print('best cv score', min_mae)

print('调参min_child_weight')
for min_child_weight in range(1, 6, 1):
    params = best_params.copy()
    params['min_child_weight'] = min_child_weight
    cv_results = xgb.cv(params=params,
                        dtrain=xgb_train,
                        num_boost_round=n_estimators,
                        nfold=5,
                        metrics=['rmse', 'mae'],
                        early_stopping_rounds=10,
                        verbose_eval=0,
                        seed=1,
                        stratified=False)
    mean_mae = cv_results['test-mae-mean'].min()
    if mean_mae <= min_mae:
        min_mae = mean_mae
        best_params['min_child_weight'] = min_child_weight
print(best_params)
print('best cv score', min_mae)

print('调参gamma')
for gamma in [i / 10.0 for i in range(0, 5)]:
    params = best_params.copy()
    params['gamma'] = gamma
    cv_results = xgb.cv(params=params,
                        dtrain=xgb_train,
                        num_boost_round=n_estimators,
                        nfold=5,
                        metrics=['rmse', 'mae'],
                        early_stopping_rounds=10,
                        verbose_eval=0,
                        seed=1,
                        stratified=False)
    mean_mae = cv_results['test-mae-mean'].min()
    if mean_mae <= min_mae:
        min_mae = mean_mae
        best_params['gamma'] = gamma
print(best_params)
print('best cv score', min_mae)

print('调参subsample和colsample_bytree')
for subsample in [i / 10.0 for i in range(0, 10)]:
    for colsample_bytree in [i / 10.0 for i in range(0, 10)]:
        params = best_params.copy()
        params['subsample'] = subsample
        params['colsample_bytree'] = colsample_bytree
        cv_results = xgb.cv(params=params,
                            dtrain=xgb_train,
                            num_boost_round=n_estimators,
                            nfold=5,
                            metrics=['rmse', 'mae'],
                            early_stopping_rounds=10,
                            verbose_eval=0,
                            seed=1,
                            stratified=False)
        mean_mae = cv_results['test-mae-mean'].min()
        if mean_mae <= min_mae:
            min_mae = mean_mae
            best_params['subsample'] = subsample
            best_params['colsample_bytree'] = colsample_bytree
print(best_params)
print('best cv score', min_mae)

print('调参alpha和lambda')
for alpha in [0, 1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.4, 0.6, 0.7, 0.9, 1.0]:
    for lambda_l2 in [0, 1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.4, 0.6, 0.7, 0.9, 1.0]:
        params = best_params.copy()
        params['alpha'] = alpha
        params['lambda'] = lambda_l2
        cv_results = xgb.cv(params=params,
                            dtrain=xgb_train,
                            num_boost_round=n_estimators,
                            nfold=5,
                            metrics=['rmse', 'mae'],
                            early_stopping_rounds=10,
                            verbose_eval=0,
                            seed=1,
                            stratified=False)
        mean_mae = cv_results['test-mae-mean'].min()
        if mean_mae <= min_mae:
            min_mae = mean_mae
            best_params['alpha'] = alpha
            best_params['lambda'] = lambda_l2
print(best_params)
print('best cv score', min_mae)

print('精调学习率，并调参n_estimators')
best_params["learning_rate"] = 0.05
cv_results = xgb.cv(params=best_params,
                    dtrain=xgb_train,
                    num_boost_round=2000,
                    nfold=5,
                    metrics=['rmse', 'mae'],
                    early_stopping_rounds=10,
                    verbose_eval=0,
                    seed=1,
                    stratified=False)
n_estimators = len(cv_results['test-mae-mean'])
min_mae = list(cv_results['test-mae-mean'])[-1]
print('best n_estimators:', n_estimators)
print('best cv score', min_mae)
print(best_params)