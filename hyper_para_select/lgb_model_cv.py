#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2020-12-30 10:50
# @Author   : Zhangyu
# @Email    : zhangycqupt@163.com
# @File     : lgb_model_cv.py
# @Software : PyCharm
# @Desc     :
"""
import pandas as pd
from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import lightgbm as lgb
import numpy as np


def data_prepare():
    """
    data_process load: http://sofasofa.io/competition.php?id=7#c1
    :return:
    """
    selected_keys = []
    category_keys = []

    trainFilePath = '../data/train.csv'
    all_data = pd.read_csv(trainFilePath)
    X, y = all_data.iloc[:, 1:len(all_data.columns) - 1], all_data['y']

    # preprocess
    X['birth_date'] = 0
    X['work_rate_att'] = X['work_rate_att'].replace('High', 2)
    X['work_rate_att'] = X['work_rate_att'].replace('Medium', 1)
    X['work_rate_att'] = X['work_rate_att'].replace('Low', 0)
    X['work_rate_def'] = X['work_rate_def'].replace('High', 2)
    X['work_rate_def'] = X['work_rate_def'].replace('Medium', 1)
    X['work_rate_def'] = X['work_rate_def'].replace('Low', 0)

    # log处理等
    # x_all = X[selected_keys + category_keys]

    # 缺失值处理
    # for c in category_keys:
    #     x_all[c].fillna(0, inplace=True)
    # x_all.fillna(-1, inplace=True)
    x_all = X

    # log处理等
    y_all = y

    X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=100)

    # testFilePath = '../data/test.csv'
    # data1 = pd.read_csv(testFilePath)
    # X_1 = data1.iloc[:, 1:]

    lgb_train = lgb.Dataset(X_train, y_train,
                            categorical_feature=category_keys,
                            free_raw_data=False)
    lgb_test = lgb.Dataset(X_test, y_test,
                           categorical_feature=category_keys,
                           free_raw_data=False)

    return [X_train, y_train, X_test, y_test, lgb_train, lgb_test]


def grid_search_cv_para_optimize():
    """
    网格搜索交叉验证调参：
    意义就是自动调参，只要把参数输进去，就能给出最优化的结果和参数。但是这个方法适合于小数据集，一旦数据的量级上去了，很难得出结果
    :return:
    """
    [X_train, y_train, X_test, y_test] = data_prepare()[0:4]
    model2 = LGBMRegressor(n_estimators=100)
    model2.fit(X_train, y_train)
    pred2 = model2.predict(X_test)
    print("mae: ", mean_absolute_error(y_test, np.expm1(pred2)))

    def bulid_modl_lgb(x_train, y_train):
        estimator = LGBMRegressor(num_leaves=127, n_estimators=150)
        param_grid = {'learning_rage': [0.01, 0.05, 0.1, 0.2]}
        gbm = GridSearchCV(estimator, param_grid)
        gbm.fit(x_train, y_train)
        return gbm

    model_lgb = bulid_modl_lgb(X_train, y_train)
    val_lgb = model_lgb.predict(X_test)
    MAE_lgb = mean_absolute_error(y_test, np.expm1(val_lgb))
    print(MAE_lgb)


def greedy_para_optimize():
    """
    贪心算法调参：拿当前对模型影响最大的参数调优，直到最优化；再拿下一个影响最大的参数调优，如此下去，直到所有的参数调整完毕。
    这个方法的缺点就是可能会调到局部最优而不是全局最优，但是省时间省力。
    :return:
    """
    [X, Y_ln] = data_prepare()[0:2]

    objective = ['regression', 'regression_l1', 'mape', 'huber', 'fair']
    num_leaves = [10, 55, 70, 100, 200]
    max_depth = [10, 55, 70, 100, 200]
    n_estimators = [200, 400, 800, 1000]
    learning_rate = [0.01, 0.05, 0.1, 0.2]

    # 先建立一个参数字典
    best_obj = dict()

    # 调objective
    for obj in objective:
        model = LGBMRegressor(objective=obj)
        score = np.mean(cross_val_score(model, X, Y_ln, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error)))
        best_obj[obj] = score

    # 上面调好之后，用上面的参数调num_leaves
    best_leaves = dict()
    for leaves in num_leaves:
        model = LGBMRegressor(objective=min(best_obj.items(), key=lambda x: x[1])[0], num_leaves=leaves)
        score = np.mean(cross_val_score(model, X, Y_ln, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error)))
        best_leaves[leaves] = score

    # 用上面两个最优参数调max_depth
    best_depth = dict()
    for depth in max_depth:
        model = LGBMRegressor(objective=min(best_obj.items(), key=lambda x: x[1])[0],
                              num_leaves=min(best_leaves.items(), key=lambda x: x[1])[0],
                              max_depth=depth)
        score = np.mean(cross_val_score(model, X, Y_ln, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error)))
        best_depth[depth] = score

    # 调n_estimators
    best_nstimators = dict()
    for nstimator in n_estimators:
        model = LGBMRegressor(objective=min(best_obj.items(), key=lambda x: x[1])[0],
                              num_leaves=min(best_leaves.items(), key=lambda x: x[1])[0],
                              max_depth=min(best_depth.items(), key=lambda x: x[1])[0],
                              n_estimators=nstimator)

        score = np.mean(cross_val_score(model, X, Y_ln, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error)))
        best_nstimators[nstimator] = score

    # 调learning_rate
    best_lr = dict()
    for lr in learning_rate:
        model = LGBMRegressor(objective=min(best_obj.items(), key=lambda x: x[1])[0],
                              num_leaves=min(best_leaves.items(), key=lambda x: x[1])[0],
                              max_depth=min(best_depth.items(), key=lambda x: x[1])[0],
                              n_estimators=min(best_nstimators.items(), key=lambda x: x[1])[0],
                              learning_rate=lr)
        score = np.mean(cross_val_score(model, X, Y_ln, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error)))
        best_lr[lr] = score

    print("best_obj:", min(best_obj.items(), key=lambda x: x[1]))
    print("best_leaves:", min(best_leaves.items(), key=lambda x: x[1]))
    print('best_depth:', min(best_depth.items(), key=lambda x: x[1]))
    print('best_nstimators: ', min(best_nstimators.items(), key=lambda x: x[1]))
    print('best_lr:', min(best_lr.items(), key=lambda x: x[1]))


def bayes_para_optimize():
    """
    pip install bayesian-optimization
    贝叶斯参数调优
    主要思想是，给定优化的目标函数(广义的函数，只需指定输入和输出即可，无需知道内部结构以及数学性质)，
    通过不断地添加样本点来更新目标函数的后验分布(高斯过程,直到后验分布基本贴合于真实分布。简单的说，就是考虑了上一次参数的信息，从而更好的调整当前的参数

    它与常规的网格搜索或者随机搜索的区别是：
    贝叶斯调参采用高斯过程，考虑之前的参数信息，不断地更新先验；网格搜索未考虑之前的参数信息
    贝叶斯调参迭代次数少，速度快；网格搜索速度慢，参数多时易导致维度爆炸
    贝叶斯调参针对非凸问题依然稳健；网格搜索针对非凸问题易得到局部最优

    使用方法：
    定义优化函数(rf_cv，在里面把优化的参数传入，然后建立模型，返回要优化的分数指标)
    定义优化参数
    开始优化（最大化分数还是最小化分数等）
    得到优化结果
    :return:
    """
    [X, Y_ln] = data_prepare()[0:2]

    # 定义优化函数
    def rf_cv(num_leaves, max_depth, subsample, min_child_samples):
        model = LGBMRegressor(objective='regression_l1',
                              num_leaves=int(num_leaves),
                              max_depth=int(max_depth),
                              subsample=subsample,
                              min_child_samples=int(min_child_samples))
        val = cross_val_score(model, X, Y_ln, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error)).mean()
        return 1 - val

    # 定义优化参数
    rf_bo = BayesianOptimization(
        rf_cv,
        {
            'num_leaves': (2, 100),
            'max_depth': (2, 100),
            'subsample': (0.1, 1),
            'min_child_samples': (2, 100)
        }
    )

    # 开始优化
    num_iter = 25
    init_points = 5
    rf_bo.maximize(init_points=init_points, n_iter=num_iter)

    # 显示优化结果
    rf_bo.res["max"]

    # 附近搜索（已经有不错的参数值的时候）
    rf_bo.explore(
        {'n_estimators': [10, 100, 200],
         'min_samples_split': [2, 10, 20],
         'max_features': [0.1, 0.5, 0.9],
         'max_depth': [5, 10, 15]
         })


def lgb_cv_para_optimize():
    """
    lgb.cv（交叉验证）调优，并结合贪心算法调优
    :return:
    """
    lgb_train = data_prepare()[5]
    min_mape = float(1)
    best_params = {}

    print("设置初始参数和学习率(=0.1),lgb.cv调优迭代步数")
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'nthread': 2,
        'learning_rate': 0.1,
        'num_leaves': 30,  # 小于2^max_depth
        'max_depth': 5,  # 根据数据量大小选择
        'bagging_fraction': 0.8,
        'feature_fraction': 0.8
    }
    cv_results = lgb.cv(
        params,
        lgb_train,
        seed=1,
        metrics=['l2', 'mape'],
        num_boost_round=1000,
        early_stopping_rounds=10,
        verbose_eval=True,
        stratified=False  # 分层（StratifiedKFold）, 不支持回归，用于分类，需禁掉
    )
    print('best n_estimators:', len(cv_results["meape-mean"]))
    print('best cv score:', cv_results["mape-mean"])

    print("调参1：提高准确率，粗调")
    n_estimators = len(cv_results["meape-mean"])
    for num_leaves in range(5, 200, 10):
        for max_depth in range(3, 9, 2):
            params['num_leaves'] = num_leaves
            params['max_depth'] = max_depth
            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=1,
                metrics=['l2', 'mape'],
                num_boost_round=n_estimators,
                early_stopping_rounds=10,
                verbose_eval=True,
                stratified=False  # 分层（StratifiedKFold）, 不支持回归，用于分类，需禁掉
            )
            mean_mape = pd.Series(cv_results['mape-mean'].min())
            if mean_mape <= min_mape:
                min_mape = mean_mape
                best_params['num_leaves'] = num_leaves
                best_params['max_depth'] = max_depth
    if 'num_leaves' and 'max_depth' in best_params.keys():
        params["num_leaves"] = best_params["num_leaves"]
        params["max_depth"] = best_params["max_depth"]
    print(best_params)

    print("调参2：提高准确率，精调")
    for num_leaves in [200, 204, 208, 212]:
        for max_depth in [7, 8, 9]:
            params['num_leaves'] = num_leaves
            params['max_depth'] = max_depth
            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=1,
                metrics=['l2', 'mape'],
                num_boost_round=n_estimators,
                early_stopping_rounds=10,
                verbose_eval=True,
                stratified=False  # 分层（StratifiedKFold）, 不支持回归，用于分类，需禁掉
            )
            mean_mape = pd.Series(cv_results['mape-mean'].min())
            if mean_mape <= min_mape:
                min_mape = mean_mape
                best_params['num_leaves'] = num_leaves
                best_params['max_depth'] = max_depth
    if 'num_leaves' and 'max_depth' in best_params.keys():
        params["num_leaves"] = best_params["num_leaves"]
        params["max_depth"] = best_params["max_depth"]
    print(best_params)

    print("调参3：降低过拟合")
    for max_bin in range(5, 256, 25):
        for min_data_in_leaf in range(1, 102, 20):
            for min_child_weight in [0.001, 0.002, 0.003, 0.004, 0.005]:
                params['max_bin'] = max_bin
                params['min_data_in_leaf'] = min_data_in_leaf
                params['min_child_weight'] = min_child_weight
                cv_results = lgb.cv(
                    params,
                    lgb_train,
                    seed=1,
                    metrics=['l2', 'mape'],
                    num_boost_round=n_estimators,
                    early_stopping_rounds=10,
                    verbose_eval=True,
                    stratified=False  # 分层（StratifiedKFold）, 不支持回归，用于分类，需禁掉
                )
                mean_mape = pd.Series(cv_results['mape-mean'].min())
                if mean_mape <= min_mape:
                    min_mape = mean_mape
                    best_params['max_bin'] = max_bin
                    best_params['min_data_in_leaf'] = min_data_in_leaf
                    best_params['min_child_weight'] = min_child_weight
    if 'max_bin' and 'max_depth' and 'min_child_weight' in best_params.keys():
        params["max_bin"] = best_params["max_bin"]
        params["min_data_in_leaf"] = best_params["min_data_in_leaf"]
        params["min_child_weight"] = best_params["min_child_weight"]
    print(best_params)

    print("调参4：降低过拟合")
    for feature_fraction in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for bagging_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
            for bagging_freq in range(0, 50, 5):
                params['feature_fraction'] = feature_fraction
                params['bagging_fraction'] = bagging_fraction
                params['bagging_freq'] = bagging_freq
                cv_results = lgb.cv(
                    params,
                    lgb_train,
                    seed=1,
                    metrics=['l2', 'mape'],
                    num_boost_round=n_estimators,
                    early_stopping_rounds=10,
                    verbose_eval=True,
                    stratified=False  # 分层（StratifiedKFold）, 不支持回归，用于分类，需禁掉
                )
                mean_mape = pd.Series(cv_results['mape-mean'].min())
                if mean_mape <= min_mape:
                    min_mape = mean_mape
                    best_params['feature_fraction'] = feature_fraction
                    best_params['bagging_fraction'] = bagging_fraction
                    best_params['bagging_freq'] = bagging_freq
    if 'feature_fraction' and 'bagging_fraction' and 'bagging_freq' in best_params.keys():
        params["feature_fraction"] = best_params["feature_fraction"]
        params["bagging_fraction"] = best_params["bagging_fraction"]
        params["bagging_freq"] = best_params["bagging_freq"]
    print(best_params)

    print("调参5：降低过拟合")
    for lambda_l1 in [0, 1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        for lambda_l2 in [0, 1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            params['lambda_l1'] = lambda_l1
            params['lambda_l2'] = lambda_l2
            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=1,
                metrics=['l2', 'mape'],
                num_boost_round=n_estimators,
                early_stopping_rounds=10,
                verbose_eval=True,
                stratified=False  # 分层（StratifiedKFold）, 不支持回归，用于分类，需禁掉
            )
            mean_mape = pd.Series(cv_results['mape-mean'].min())
            if mean_mape <= min_mape:
                min_mape = mean_mape
                best_params['lambda_l1'] = lambda_l1
                best_params['lambda_l2'] = lambda_l2
    if 'lambda_l1' and 'lambda_l2' in best_params.keys():
        params["lambda_l1"] = best_params["lambda_l1"]
        params["lambda_l2"] = best_params["lambda_l2"]
    print(best_params)

    print("调参6：降低过拟合")
    for min_split_gain in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        params['min_split_gain'] = min_split_gain
        cv_results = lgb.cv(
            params,
            lgb_train,
            seed=1,
            metrics=['l2', 'mape'],
            num_boost_round=n_estimators,
            early_stopping_rounds=10,
            verbose_eval=True,
            stratified=False  # 分层（StratifiedKFold）, 不支持回归，用于分类，需禁掉
        )
        mean_mape = pd.Series(cv_results['mape-mean'].min())
        if mean_mape <= min_mape:
            min_mape = mean_mape
            best_params['min_split_gain'] = min_split_gain
    if 'min_split_gain' in best_params.keys():
        params["min_split_gain"] = best_params["min_split_gain"]
    print(best_params)

    print("学习率调优，细水长流提高精度")
    params["learning_rate"] = 0.005
    n_estimators = 2000
    cv_results = lgb.cv(
        params,
        lgb_train,
        seed=1,
        metrics=['l2', 'mape'],
        num_boost_round=n_estimators,
        early_stopping_rounds=10,
        verbose_eval=True,
        stratified=False  # 分层（StratifiedKFold）, 不支持回归，用于分类，需禁掉
    )
    print('best n_estimators:', len(cv_results["meape-mean"]))
    print('best cv score:', cv_results["mape-mean"])


if __name__ == "__main__":
    # data = data_prepare()
    lgb_cv_para_optimize()
    pass
