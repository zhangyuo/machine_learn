#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019-03-26 14:43
@Author  : zhangyu
@Contact : zhangycqupt@163.com
@File    : kernel_ridge_regression.py
@Software: PyCharm
@Site    : https://github.com/zhangyuo
"""

"""
kernel ridge regression(KRR):核岭回归，在岭回归的基础上加上了核技巧。也叫做最小二乘SVM(LSSVM)，损失函数是最小二乘法的SVM。
线性核岭回归模型中，模型复杂度与特征维度d有关，而非线性核岭回归中，模型复杂度与样本数N有关。通常样本数低于1000具有较好的实用性。

Support Vectors Regression(SVR):支持向量回归，是普通的软间隔SVM的回归模型。其rfa值大部分为0，支持向量较少，不是稠密的。
参考：https://sklearn.org/modules/classes.html
svm.LinearSVR([epsilon, tol, C, loss, …])	Linear Support Vector Regression.
svm.NuSVR([nu, C, kernel, degree, gamma, …])	Nu Support Vector Regression.
svm.SVR([kernel, degree, gamma, coef0, tol, …])	Epsilon-Support Vector Regression.
"""

from sklearn.kernel_ridge import KernelRidge
import numpy as np


def kernel_ridge_regression():
    n_samples, n_features = 10, 5
    rng = np.random.RandomState(0)
    y = rng.randn(n_samples)
    X = rng.randn(n_samples, n_features)
    clf = KernelRidge(alpha=1.0)
    clf.fit(X, y)
    print(clf.alpha)
    print(clf.get_params())


if __name__ == '__main__':
    # kernel_ridge_regression()
    # from __future__ import division
    import time

    import numpy as np

    from sklearn.svm import SVR
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import learning_curve
    from sklearn.kernel_ridge import KernelRidge
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(0)

    # #############################################################################
    # Generate sample data_process
    X = 5 * rng.rand(10000, 1)
    y = np.sin(X).ravel()

    # Add noise to targets
    y[::5] += 3 * (0.5 - rng.rand(X.shape[0] // 5))

    X_plot = np.linspace(0, 5, 100000)[:, None]

    # #############################################################################
    # Fit regression model
    train_size = 100
    svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                       param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                   "gamma": np.logspace(-2, 2, 5)})

    kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                      param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                                  "gamma": np.logspace(-2, 2, 5)})

    t0 = time.time()
    svr.fit(X[:train_size], y[:train_size])
    svr_fit = time.time() - t0
    print("SVR complexity and bandwidth selected and model fitted in %.3f s"
          % svr_fit)

    t0 = time.time()
    kr.fit(X[:train_size], y[:train_size])
    kr_fit = time.time() - t0
    print("KRR complexity and bandwidth selected and model fitted in %.3f s"
          % kr_fit)

    sv_ratio = svr.best_estimator_.support_.shape[0] / train_size
    print("Support vector ratio: %.3f" % sv_ratio)

    t0 = time.time()
    y_svr = svr.predict(X_plot)
    svr_predict = time.time() - t0
    print("SVR prediction for %d inputs in %.3f s"
          % (X_plot.shape[0], svr_predict))

    t0 = time.time()
    y_kr = kr.predict(X_plot)
    kr_predict = time.time() - t0
    print("KRR prediction for %d inputs in %.3f s"
          % (X_plot.shape[0], kr_predict))

    # #############################################################################
    # Look at the results
    sv_ind = svr.best_estimator_.support_
    plt.scatter(X[sv_ind], y[sv_ind], c='r', s=50, label='SVR support vectors',
                zorder=2, edgecolors=(0, 0, 0))
    plt.scatter(X[:100], y[:100], c='k', label='data_process', zorder=1,
                edgecolors=(0, 0, 0))
    plt.plot(X_plot, y_svr, c='r',
             label='SVR (fit: %.3fs, predict: %.3fs)' % (svr_fit, svr_predict))
    plt.plot(X_plot, y_kr, c='g',
             label='KRR (fit: %.3fs, predict: %.3fs)' % (kr_fit, kr_predict))
    plt.xlabel('data_process')
    plt.ylabel('target')
    plt.title('SVR versus Kernel Ridge')
    plt.legend()

    # Visualize training and prediction time
    plt.figure()

    # Generate sample data_process
    X = 5 * rng.rand(10000, 1)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - rng.rand(X.shape[0] // 5))
    sizes = np.logspace(1, 4, 7).astype(np.int)
    for name, estimator in {"KRR": KernelRidge(kernel='rbf', alpha=0.1,
                                               gamma=10),
                            "SVR": SVR(kernel='rbf', C=1e1, gamma=10)}.items():
        train_time = []
        test_time = []
        for train_test_size in sizes:
            t0 = time.time()
            estimator.fit(X[:train_test_size], y[:train_test_size])
            train_time.append(time.time() - t0)

            t0 = time.time()
            estimator.predict(X_plot[:1000])
            test_time.append(time.time() - t0)

        plt.plot(sizes, train_time, 'o-', color="r" if name == "SVR" else "g",
                 label="%s (train)" % name)
        plt.plot(sizes, test_time, 'o--', color="r" if name == "SVR" else "g",
                 label="%s (test)" % name)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Train size")
    plt.ylabel("Time (seconds)")
    plt.title('Execution Time')
    plt.legend(loc="best")

    # Visualize learning curves
    plt.figure()

    svr = SVR(kernel='rbf', C=1e1, gamma=0.1)
    kr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1)
    train_sizes, train_scores_svr, test_scores_svr = \
        learning_curve(svr, X[:100], y[:100], train_sizes=np.linspace(0.1, 1, 10),
                       scoring="neg_mean_squared_error", cv=10)
    train_sizes_abs, train_scores_kr, test_scores_kr = \
        learning_curve(kr, X[:100], y[:100], train_sizes=np.linspace(0.1, 1, 10),
                       scoring="neg_mean_squared_error", cv=10)

    plt.plot(train_sizes, -test_scores_svr.mean(1), 'o-', color="r",
             label="SVR")
    plt.plot(train_sizes, -test_scores_kr.mean(1), 'o-', color="g",
             label="KRR")
    plt.xlabel("Train size")
    plt.ylabel("Mean Squared Error")
    plt.title('Learning curves')
    plt.legend(loc="best")

    plt.show()

