#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/3/22 16:52
@Author  : Zhangyu
@Email   : zhangycqupt@163.com
@File    : linear_regression.py
@Software: PyCharm
@Github  : zhangyuo
"""

"""
线性回归模型：经典模型包括最小二乘回归、岭回归、lasso回归
线性回归：线性最小二乘回归
岭回归：线性最小二乘回归+L2范数，当数据有多重共现时实用，即自变量存在高相关性
lasso回归：线性最小二乘回归+L1范数
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


def ordinary_least_squares():
    """
    最小二乘线性回归-单变量、取糖尿病数据集一维特征
    容易受随机噪声的影响，导致误差加大
    If X is a matrix of size (n, p) this method has a cost of O(n*p*p)
    :return:
    """
    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()

    # Use only one feature for linear regression
    diabetes_X = diabetes.data[:, np.newaxis, 2]
    # diabetes_X = diabetes.data_process

    # Split the data_process into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    # diabetes_X_train = diabetes_X[:-20:]
    # diabetes_X_test = diabetes_X[:-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

    # Plot outputs
    plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
    plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()


def ridge_regression():
    """
    岭回归
    复杂度同最小二乘法
    由于增加了系数w的l2范数，有效降低了过拟合风险
    计算量较lasso大
    :return:
    """
    # reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3)
    reg = linear_model.Ridge(alpha=.5)
    reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
    # Regularization strength
    # print(reg.alpha_)
    print(reg.coef_)


def lasso_regression():
    """
    lasso回归
    由于增加了系数w的l1范数，可降低过拟合分享
    lasso回归能使得损失函数中许多theta变为0，计算量较岭回归小
    :return:
    """
    reg = linear_model.Lasso(alpha=0.1)
    reg.fit([[0, 0], [1, 1]], [0, 1])
    a = reg.predict([[1, 1]])
    print(a)
    print(reg.coef_)


if __name__ == "__main__":
    # ordinary_least_squares()
    # ridge_regression()
    lasso_regression()
