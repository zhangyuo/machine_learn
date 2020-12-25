#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019-04-03 17:22
@Author  : zhangyu
@Contact : zhangycqupt@163.com
@File    : svms.py
@Software: PyCharm
@Site    : https://github.com/zhangyuo
"""

"""
svm:支持向量机，可用于分类、回归、异常值检测
https://scikit-learn.org/stable/modules/svm.html

优点：
高维空间有效、
维度数量大于样本数量有效、
仅使用训练数据的一个子集（支持向量），因此是内存友好型的算法、
仅使用训练数据的一个子集（支持向量），因此是内存友好型的算法

缺点：
在数据维度（特征个数）多于样本数很多的时候，通常只能训练出一个表现很差的模型。
SVM不支持直接进行概率估计，Scikit中使用很耗费资源的5折交叉检验来估计概率。

核心思想：
支持向量机算法的核心思想在于找到那个可以最稳健的将样本进行分类的间隔超平面，其稳健性来源于在尽量确保分类正确的前提下，
会寻找到可以最大化位于超平面两侧的距离超平面最近的点的间隔 Margin，这些离超平面最近的点被称为支持向量 Support Vector。

重要参数：
C：误差项的惩罚系数，用于调节决策边界的平滑程度和分类准确性，C 越大则对于误差的惩罚越大，分类正确的点越多，决策边界越倾向于过拟合
Kernel Functions：对于非线性可分的样本，我们可以通过引入核函数来增加已有的数据的维度，使得其可以被投影到另外一个空间，进而便于实现分割或分类
gamma: 核函数的参数，gamma 越大则分类越准确，例如在高斯核 Radial Basis Function with a Gaussian Kernel 中，我们可以令 z = e-γ(x2+y2) 来实现维度扩增

Logistic Regression VS SVMs
当特征的数量 n 与样本的数量 m 相比较大时，如当 n = 10,000 而 m = 10 - 1000，优先选择逻辑回归，或采用线性核 SVM
当特征的数量 n 较小而 m 的数量中等时，如 n = 1 - 1000 而 m = 10 - 10,000 时，采用 Gaussian 核的 SVM
当特征的数量 n 较小而 m 很大时，如 n = 1 - 1000 而 m = 50,000+，由于采用高斯核计算量较大，此时可以考虑增加特征，并采用逻辑回归或线性核 SVM

机器学习中的算法选择
机器学习的各种算法和模型实际上提供了一个工具组合，我们在实际使用中可能很难仅凭直觉确定哪个模型会给出更好的拟合结果，因此最可靠的方法就是尝试利用训练数据训练多个模型，对应每一个模型选择不同的超参数，在此基础上通过 K-Fold Cross Validation 来选出那个给予最高准确率或评价指标的模型，再通过测试数据进行模型性能评估。
当模型具有多个可选超参数时，例如在 SVM 中，在实际使用中 C 和 gamma 的选择会很大程度上影响分类的结果和决策边界的设定，在 Scikit-Learn 中可以通过采用 Grid Search 的方法来实现最优参数的选择。

"""
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import datasets


def svc():
    """
    二分类
    classifier = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)
    """

    data_set = datasets.load_breast_cancer()
    X = data_set.data
    y = data_set.target
    # Split data_process to train and test on 80-20 ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Parameter grid search
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

    # Make a grid search classifier
    grid_classifier = GridSearchCV(svm.SVC(), param_grid, verbose=1)

    # Train the classifier
    grid_classifier.fit(X_train, y_train)

    # See what are the best parameters
    print("Best parameters:\n", grid_classifier.best_params_)
    print("Best estimations:\n", grid_classifier.best_estimator_)
    print("Best scores:\n", grid_classifier.score(X_test, y_test))

    # test
    print(grid_classifier.predict(X_test))
    print(y_test)


def svc_multi():
    """
    多分类
    n_class * (n_class - 1) / 2 classifiers - ovo
    n_class classifers - ovr
    :return:
    """
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # Split data_process to train and test on 80-20 ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf = svm.SVC(gamma='scale', decision_function_shape='ovo', probability=True)  # enable probability
    clf.fit(X_train, y_train)
    # parameters
    print("support vectors\n", clf.support_vectors_)
    print("indices of support vectors\n", clf.support_)
    print("number of support vectors for each class\n", clf.n_support_)

    dec = clf.decision_function([[1, 2, 3, 4]])
    print("num of classes\n", dec.shape[1])

    # test
    print(clf.predict(X_test))
    print(clf.predict_proba(X_test))
    # print(clf.predict_log_proba(X_test))
    print(y_test)


if __name__ == '__main__':
    # svc()
    svc_multi()
