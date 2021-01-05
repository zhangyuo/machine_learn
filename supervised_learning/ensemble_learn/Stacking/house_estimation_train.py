#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2021-01-05 16:00
# @Author   : Zhangyu
# @Email    : zhangycqupt@163.com
# @File     : house_estimation_train.py
# @Software : PyCharm
# @Desc     :
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from tqdm import tqdm
from xgboost import XGBRegressor

x = load_boston().data
y = load_boston().target

x = MinMaxScaler().fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=100)
kf = KFold(n_splits=5, random_state=100)
n_train = xtrain.shape[0]
n_test = xtest.shape[0]
print("训练样本个数：", n_train)
print("测试样本个数：", n_test)

# meta model choose，可以是同质学习器，也可以是异质学习器，但一般来说要保证不同学习器具有较为一致的结果（相差不能太大）
models = [
    RandomForestRegressor(n_estimators=300, random_state=100),
    ExtraTreesRegressor(n_estimators=300, random_state=100, n_jobs=-1),
    GradientBoostingRegressor(n_estimators=300, random_state=100),
    XGBRegressor(n_estimators=300),
    LGBMRegressor(n_estimators=300, random_state=100, n_jobs=-1),
    RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 10, 20, 30, 50]),
    LinearRegression(),
    SVR(kernel='rbf'),
    SVR(kernel='linear')
]


def get_oof(model, x_train, y_train, x_test):
    """

    :param model: meta model
    :param x_train: train data x
    :param y_train: train data y
    :param x_test: test data x
    :return:
    """
    oof_train = np.zeros((n_train,))  # 构造一个1*354的一维0矩阵
    oof_test = np.zeros((n_test,))  # 构造一个1*152的一维0矩阵
    oof_test_skf = np.zeros((5, n_test))  # 5*152
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        kf_x_train = x_train[train_index]  # 每一折训练283个样本的X
        kf_y_train = y_train[train_index]  # 每一折训练283个样本的Y
        kf_x_test = x_train[test_index]  # 每一折71个测试样本的X
        model = model.fit(kf_x_train, kf_y_train)
        oof_train[test_index] = model.predict(kf_x_test)  # 每次产生71个预测值，最终5折后堆叠成为1*354个训练样本的测试值
        oof_test_skf[i, :] = model.predict(x_test)  # 每次生成1*152的测试集预测值，5折后填满5*152的预测值矩阵
    oof_test[:] = oof_test_skf.mean(axis=0)  # 把 测试集的五次预测结果，求平均，形成一次预测结果
    return oof_train, oof_test  # 第一个返回值为第二层模型训练集xtrain的特征，1*354；第二个返回值为第一层模型对测试集数据的预测1*152，作为第二层模型的训练集xtest


n_models = len(models)
print("元模型个数：", n_models)
xtrain_new = np.zeros((n_train, n_models))
xtest_new = np.zeros((n_test, n_models))
for i, regressor in tqdm(enumerate(models), desc="offset:"):
    xtrain_new[:, i], xtest_new[:, i] = get_oof(regressor, xtrain, ytrain, xtest)

# 基于元模型和元数据，训练第二层模型
reg = LinearRegression()
reg.fit(xtrain_new, ytrain)
# result = reg.predict(xtest_new)
score = reg.score(xtest_new, ytest)
print(score)
