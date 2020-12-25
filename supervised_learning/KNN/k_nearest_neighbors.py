#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019-04-04 16:39
@Author  : zhangyu
@Contact : zhangycqupt@163.com
@File    : nearest_neighbors.py
@Software: PyCharm
@Site    : https://github.com/zhangyuo
"""

"""
有监督的最近邻方法包括：离散数据的分类、连续数据的回归。
https://www.jianshu.com/p/778d39132d66

最近邻分类
它是一种基于实例的学习或者非泛化学习（a type of instance-based learning or non-generalizing learning），它不试图构建一个通用的内部模型，只是简单的存储训练数据的实例。通过每个点的最近邻的简单多数投票来分类，一个查询点被分配给它的最近邻中最有代表性的那个数据类。
scikit-learn 包括两种最近邻分类器：KNeighborsClassifier 和 RadiusNeighborsClassifier。KNeighborsClassifier是比较常用的一种，一般的，一个比较大的k值能抑制异常值的影响，但是也让分类边界的区分性下降。
如果是不均匀采样数据，RadiusNeighborsClassifier中的基于半径的最近邻分类是一个比较好的选择。用户指定一个固定的半径 r ，那些稀疏区域中的点使用较少的最近邻去分类，在高维参数空间，由于所谓的“维灾难”，这种方法就变得不那么有效。
基本的最近邻分类器使用一致权重，即采用最近邻数据的简单多数表决。在某些情况下，更近的数据点对模型的贡献更大，可以通过参数 weights 来设置。默认情况下，weights=‘uniform’，所有邻居的权重是相同的。weights=‘distance’ ，权重正比于到查询点的距离的倒数。另外，用户也可以自定义函数计算权重。

最近邻回归
当数据标签是离散的而不是连续的，可以使用最近邻回归方法。查询点的标签通过最近邻点的平均标签计算。
scikit-learn有两种最近邻回归方法：KNeighborsRegressor 和 RadiusNeighborsRegressor。
"""


def knn_c():
    """
    k近邻分类
    :return:
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from sklearn import neighbors, datasets

    n_neighbors = 15

    # import some data_process to play with
    iris = datasets.load_iris()

    # we only take the first two features. We could avoid this ugly
    # slicing by using a two-dim dataset
    X = iris.data[:, :2]
    y = iris.target

    h = .02  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    for weights in ['uniform', 'distance']:
        # we create an instance of Neighbours Classifier and fit the data_process.
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X, y)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                    edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification (k = %i, weights = '%s')"
                  % (n_neighbors, weights))

    plt.show()


def knn_r():
    """
    k近邻回归
    :return:
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import neighbors

    np.random.seed(0)
    X = np.sort(5 * np.random.rand(40, 1), axis=0)
    T = np.linspace(0, 5, 500)[:, np.newaxis]
    y = np.sin(X).ravel()

    # Add noise to targets
    y[::5] += 1 * (0.5 - np.random.rand(8))

    # #############################################################################
    # Fit regression model
    n_neighbors = 5

    for i, weights in enumerate(['uniform', 'distance']):
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
        y_ = knn.fit(X, y).predict(T)

        plt.subplot(2, 1, i + 1)
        plt.scatter(X, y, c='k', label='data_process')
        plt.plot(T, y_, c='g', label='prediction')
        plt.axis('tight')
        plt.legend()
        plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                    weights))

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # knn_c()
    knn_r()
