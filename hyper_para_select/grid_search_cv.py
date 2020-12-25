#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2020-10-15 20:23
# @Author   : Zhangyu
# @Email    : zhangycqupt@163.com
# @File     : grid_search_cv.py
# @Software : PyCharm
# @Desc     :
"""

from sklearn.model_selection import train_test_split
from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# 使用sklearn库中自带的iris数据集作为示例
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)  # 分割数据集


# GridSearchCV方法，自动对输入的参数进行排列组合，并一一测试，从中选出最优的一组参数

# param_grid-设置参数调整的范围及配置，这里的参数都是人为指定的。值为字典或者列表，即需要最优化的参数的取值
param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]
# scoring-准确度评价标准，默认None,这时需要使用score函数；或者如scoring='roc_auc'，根据所选模型不同，评价准则不同
"""
1-Classification
‘accuracy’
metrics.accuracy_score
‘balanced_accuracy’
metrics.balanced_accuracy_score
‘average_precision’
metrics.average_precision_score
‘neg_brier_score’
metrics.brier_score_loss
‘f1’
metrics.f1_score
for binary targets
‘f1_micro’
metrics.f1_score
micro-averaged
‘f1_macro’
metrics.f1_score
macro-averaged
‘f1_weighted’
metrics.f1_score
weighted average
‘f1_samples’
metrics.f1_score
by multilabel sample
‘neg_log_loss’
metrics.log_loss
requires predict_proba support
‘precision’ etc.
metrics.precision_score
suffixes apply as with ‘f1’
‘recall’ etc.
metrics.recall_score
suffixes apply as with ‘f1’
‘jaccard’ etc.
metrics.jaccard_score
suffixes apply as with ‘f1’
‘roc_auc’
metrics.roc_auc_score
‘roc_auc_ovr’
metrics.roc_auc_score
‘roc_auc_ovo’
metrics.roc_auc_score
‘roc_auc_ovr_weighted’
metrics.roc_auc_score
‘roc_auc_ovo_weighted’
metrics.roc_auc_score

2-Clustering
‘adjusted_mutual_info_score’
metrics.adjusted_mutual_info_score
‘adjusted_rand_score’
metrics.adjusted_rand_score
‘completeness_score’
metrics.completeness_score
‘fowlkes_mallows_score’
metrics.fowlkes_mallows_score
‘homogeneity_score’
metrics.homogeneity_score
‘mutual_info_score’
metrics.mutual_info_score
‘normalized_mutual_info_score’
metrics.normalized_mutual_info_score
‘v_measure_score’
metrics.v_measure_score

3-Regression
‘explained_variance’
metrics.explained_variance_score
‘max_error’
metrics.max_error
‘neg_mean_absolute_error’
metrics.mean_absolute_error
‘neg_mean_squared_error’
metrics.mean_squared_error
‘neg_root_mean_squared_error’
metrics.mean_squared_error
‘neg_mean_squared_log_error’
metrics.mean_squared_log_error
‘neg_median_absolute_error’
metrics.median_absolute_error
‘r2’
metrics.r2_score
‘neg_mean_poisson_deviance’
metrics.mean_poisson_deviance
‘neg_mean_gamma_deviance’
metrics.mean_gamma_deviance

"""

# 将超参数配置及模型放入GridSearch中进行自动搜索
svm_model = svm.SVC()

clf = GridSearchCV(svm_model, param_grid, cv=5)
clf.fit(X_train, y_train)

# 获取选择的最优模型
best_model = clf.best_estimator_

# 查看选择的最优超参数配置
print(clf.best_params_)

# 预测
y_pred = best_model.predict(X_test)
print('accuracy', accuracy_score(y_test, y_pred))
