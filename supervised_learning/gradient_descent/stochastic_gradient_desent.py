#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019-04-04 15:48
@Author  : zhangyu
@Contact : zhangycqupt@163.com
@File    : stochastic_gradient_desent.py
@Software: PyCharm
@Site    : https://github.com/zhangyuo
"""

"""
sgd：随机梯度下降
主要应用在大规模稀疏数据问题上，经常用在文本分类及自然语言处理。假如数据是稀疏的，
该模块的分类器可轻松解决如下问题：超过10^5 的训练样本、超过 10^5 的features。利用梯度来求解参数。

http://scikit-learn.org/stable/supervised_learning.html#supervised-learning
随机梯度下降分类器和回归器，还在奇怪，随机梯度下降难道不是一种求解参数的方法，难道可以用来做分类和回归？
原来，随机梯度下降分类器并不是一个独立的算法，而是一系列利用随机梯度下降求解参数的算法的集合。
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import SGDClassifier


def sgd_binary_classify():
    """
    二分类
    :return:
    """
    X = [[0., 0.], [1., 1.]]
    y = [0, 1]
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    clf.fit(X, y)

    print("model parameters:\n", clf.coef_)
    print("the intercept (aka offset or bias:\n", clf.intercept_)
    # test
    print(clf.predict([[2., 2.]]))


def sgd_multi_classify():
    """
    ova
    多分类
    :return:
    """
    # import some data_process to play with
    iris = datasets.load_iris()

    # we only take the first two features. We could
    # avoid this ugly slicing by using a two-dim dataset
    X = iris.data[:, :2]
    y = iris.target
    colors = "bry"

    # shuffle
    idx = np.arange(X.shape[0])
    np.random.seed(13)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    h = .02  # step size in the mesh

    clf = SGDClassifier(alpha=0.001, max_iter=100).fit(X, y)

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis('tight')

    # Plot also the training points
    for i, color in zip(clf.classes_, colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.Paired, edgecolor='black', s=20)
    plt.title("Decision surface of multi-class SGD")
    plt.axis('tight')

    # Plot the three one-against-all classifiers
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    coef = clf.coef_
    intercept = clf.intercept_

    def plot_hyperplane(c, color):
        def line(x0):
            return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]

        plt.plot([xmin, xmax], [line(xmin), line(xmax)],
                 ls="--", color=color)

    for i, color in zip(clf.classes_, colors):
        plot_hyperplane(i, color)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # sgd_binary_classify()
    sgd_multi_classify()
