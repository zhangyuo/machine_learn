#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2020-10-15 20:49
# @Author   : Zhangyu
# @Email    : zhangycqupt@163.com
# @File     : train.py
# @Software : PyCharm
# @Desc     :
"""
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
import xgboost as xgb
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
# GridSearchCV使用时参数配置:
(1)param_grid：值为字典或者列表，即需要最优化的参数的取值。比如：cvparams = {'n_estimators': [550, 575, 600, 650, 675]}
(2)scoring : 准确度评价标准，默认None，这时需要使用score函数；或者如scoring='roc_auc'，根据所选模型不同，评价准则不同。
具体选择函数对应不同含义见链接：https://scikit-learn.org/stable/modules/model_evaluation.html
具体分为三大类：classification/clustering/regression
"""
#  本次使用xgb进行回归预测，使用score = 'r2'函数  ## metrics.r2_score
# 初始化xgb参数
# xgb参数列表：https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-tweedie-regression-objective-reg-tweedie
learning_rate = 0.1
n_estimators = 500
max_depth = 5
minchildweight = 1
subsample = 0.8
colsample_bytree = 0.8
gamma = 0
reg_alpha = 0
reg_lambda = 1

# data_process load: http://sofasofa.io/competition.php?id=7#c1
trainFilePath = '../data/train.csv'
testFilePath = '../data/test.csv'
data = pd.read_csv(trainFilePath)
X, y = data.iloc[:, 1:len(data.columns) - 1], data['y']
X['birth_date'] = 0
X['work_rate_att'] = X['work_rate_att'].replace('High', 2)
X['work_rate_att'] = X['work_rate_att'].replace('Medium', 1)
X['work_rate_att'] = X['work_rate_att'].replace('Low', 0)
X['work_rate_def'] = X['work_rate_def'].replace('High', 2)
X['work_rate_def'] = X['work_rate_def'].replace('Medium', 1)
X['work_rate_def'] = X['work_rate_def'].replace('Low', 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
data1 = pd.read_csv(testFilePath)
X_1 = data1.iloc[:, 1:]

# 1、选择最佳迭代次数：n_estimators
cv_params = {'n_estimators': [400, 500, 600, 700, 800]}
other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

model = xgb.XGBRegressor(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=1)
optimized_GBM.fit(X_train, y_train)
evalute_result = optimized_GBM.cv_results_
means = evalute_result['mean_test_score']
arams = evalute_result['params']
print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

# 运行后的结果为：
# [Parallel(n_jobs=4)]: Done  25 out of  25 | elapsed:  1.5min finished
# 每轮迭代运行结果:[mean: 0.94051, std: 0.01244, params: {'n_estimators': 400}, mean: 0.94057, std: 0.01244, params: {'n_estimators': 500}, mean: 0.94061, std: 0.01230, params: {'n_estimators': 600}, mean: 0.94060, std: 0.01223, params: {'n_estimators': 700}, mean: 0.94058, std: 0.01231, params: {'n_estimators': 800}]
# 参数的最佳取值：{'n_estimators': 600}
# 最佳模型得分:0.9406056804545407
# 由输出结果可知最佳迭代次数为600次。但是，我们还不能认为这是最终的结果，由于设置的间隔太大，所以，我又测试了一组参数，这次粒度小一些：

cv_params = {'n_estimators': [550, 575, 600, 650, 675]}
other_params = {'learning_rate': 0.1, 'n_estimators': 600, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
# 运行后的结果为：
# [Parallel(n_jobs=4)]: Done  25 out of  25 | elapsed:  1.5min finished
# 每轮迭代运行结果:[mean: 0.94065, std: 0.01237, params: {'n_estimators': 550}, mean: 0.94064, std: 0.01234, params: {'n_estimators': 575}, mean: 0.94061, std: 0.01230, params: {'n_estimators': 600}, mean: 0.94060, std: 0.01226, params: {'n_estimators': 650}, mean: 0.94060, std: 0.01224, params: {'n_estimators': 675}]
# 参数的最佳取值：{'n_estimators': 550}
# 最佳模型得分:0.9406545392685364
# 果不其然，最佳迭代次数变成了550。有人可能会问，那还要不要继续缩小粒度测试下去呢？这个我觉得可以看个人情况，如果你想要更高的精度，当然是粒度越小，结果越准确，大家可以自己慢慢去调试，我在这里就不一一去做了。

# 2、接下来要调试的参数是minchildweight以及max_depth：
# 注意：每次调完一个参数，要把 other_params对应的参数更新为最优值。

cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
other_params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
# 运行后的结果为：
# [Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:  1.7min
# [Parallel(n_jobs=4)]: Done 192 tasks      | elapsed: 12.3min
# [Parallel(n_jobs=4)]: Done 240 out of 240 | elapsed: 17.2min finished
# 每轮迭代运行结果:[mean: 0.93967, std: 0.01334, params: {'min_child_weight': 1, 'max_depth': 3}, mean: 0.93826, std: 0.01202, params: {'min_child_weight': 2, 'max_depth': 3}, mean: 0.93739, std: 0.01265, params: {'min_child_weight': 3, 'max_depth': 3}, mean: 0.93827, std: 0.01285, params: {'min_child_weight': 4, 'max_depth': 3}, mean: 0.93680, std: 0.01219, params: {'min_child_weight': 5, 'max_depth': 3}, mean: 0.93640, std: 0.01231, params: {'min_child_weight': 6, 'max_depth': 3}, mean: 0.94277, std: 0.01395, params: {'min_child_weight': 1, 'max_depth': 4}, mean: 0.94261, std: 0.01173, params: {'min_child_weight': 2, 'max_depth': 4}, mean: 0.94276, std: 0.01329...]
# 参数的最佳取值：{'min_child_weight': 5, 'max_depth': 4}
# 最佳模型得分:0.94369522247392
# 由输出结果可知参数的最佳取值：{'min_child_weight': 5, 'max_depth': 4}。（代码输出结果被我省略了一部分，因为结果太长了，以下也是如此）

# 3、接着我们就开始调试参数：gamma：

cv_params = {'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
other_params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 4, 'min_child_weight': 5, 'seed': 0,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
# 运行后的结果为：
# [Parallel(n_jobs=4)]: Done  30 out of  30 | elapsed:  1.5min finished
# 每轮迭代运行结果:[mean: 0.94370, std: 0.01010, params: {'gamma': 0.1}, mean: 0.94370, std: 0.01010, params: {'gamma': 0.2}, mean: 0.94370, std: 0.01010, params: {'gamma': 0.3}, mean: 0.94370, std: 0.01010, params: {'gamma': 0.4}, mean: 0.94370, std: 0.01010, params: {'gamma': 0.5}, mean: 0.94370, std: 0.01010, params: {'gamma': 0.6}]
# 参数的最佳取值：{'gamma': 0.1}
# 最佳模型得分:0.94369522247392
# 由输出结果可知参数的最佳取值：{'gamma': 0.1}。

# 4、接着是subsample以及colsample_bytree：

cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
other_params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 4, 'min_child_weight': 5, 'seed': 0,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.1, 'reg_alpha': 0, 'reg_lambda': 1}
# 运行后的结果显示参数的最佳取值：{'subsample': 0.7,'colsample_bytree': 0.7}

# 5、紧接着就是：regalpha以及reglambda：

cv_params = {'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]}
other_params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 4, 'min_child_weight': 5, 'seed': 0,
                'subsample': 0.7, 'colsample_bytree': 0.7, 'gamma': 0.1, 'reg_alpha': 0, 'reg_lambda': 1}
# 运行后的结果为：
# [Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:  2.0min
# [Parallel(n_jobs=4)]: Done 125 out of 125 | elapsed:  5.6min finished
# 每轮迭代运行结果:[mean: 0.94169, std: 0.00997, params: {'reg_alpha': 0.01, 'reg_lambda': 0.01}, mean: 0.94112, std: 0.01086, params: {'reg_alpha': 0.01, 'reg_lambda': 0.05}, mean: 0.94153, std: 0.01093, params: {'reg_alpha': 0.01, 'reg_lambda': 0.1}, mean: 0.94400, std: 0.01090, params: {'reg_alpha': 0.01, 'reg_lambda': 1}, mean: 0.93820, std: 0.01177, params: {'reg_alpha': 0.01, 'reg_lambda': 100}, mean: 0.94194, std: 0.00936, params: {'reg_alpha': 0.05, 'reg_lambda': 0.01}, mean: 0.94136, std: 0.01122, params: {'reg_alpha': 0.05, 'reg_lambda': 0.05}, mean: 0.94164, std: 0.01120...]
# 参数的最佳取值：{'reg_alpha': 1, 'reg_lambda': 1}
# 最佳模型得分:0.9441561344357595
# 由输出结果可知参数的最佳取值：{'reg_alpha': 1, 'reg_lambda': 1}。

# 6、最后就是learning_rate，一般这时候要调小学习率来测试：

cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
other_params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 4, 'min_child_weight': 5, 'seed': 0,
                'subsample': 0.7, 'colsample_bytree': 0.7, 'gamma': 0.1, 'reg_alpha': 1, 'reg_lambda': 1}


# 运行后的结果为：
# [Parallel(n_jobs=4)]: Done  25 out of  25 | elapsed:  1.1min finished
# 每轮迭代运行结果:[mean: 0.93675, std: 0.01080, params: {'learning_rate': 0.01}, mean: 0.94229, std: 0.01138, params: {'learning_rate': 0.05}, mean: 0.94110, std: 0.01066, params: {'learning_rate': 0.07}, mean: 0.94416, std: 0.01037, params: {'learning_rate': 0.1}, mean: 0.93985, std: 0.01109, params: {'learning_rate': 0.2}]
# 参数的最佳取值：{'learning_rate': 0.1}
# 最佳模型得分:0.9441561344357595
# 由输出结果可知参数的最佳取值：{'learning_rate': 0.1}。

# 我们可以很清楚地看到，随着参数的调优，最佳模型得分是不断提高的，这也从另一方面验证了调优确实是起到了一定的作用。不过，我们也可以注意到，其实最佳分数并没有提升太多。提醒一点，这个分数是根据前面设置的得分函数算出来的，即：
# optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)中的scoring='r2'。在实际情境中，我们可能需要利用各种不同的得分函数来评判模型的好坏。

# 最后，我们把得到的最佳参数组合扔到模型里训练，就可以得到预测的结果了：
def trainandTest(X_train, y_train, X_test):
    # XGBoost训练过程，下面的参数就是刚才调试出来的最佳参数组合
    model = xgb.XGBRegressor(learning_rate=0.1, n_estimators=550, max_depth=4, min_child_weight=5, seed=0,
                             subsample=0.7, colsample_bytree=0.7, gamma=0.1, reg_alpha=1, reg_lambda=1)
    model.fit(X_train, y_train)

    # 对测试集进行预测
    ans = model.predict(X_test)

    ans_len = len(ans)
    id_list = np.arange(10441, 17441)
    data_arr = []
    for row in range(0, ans_len):
        data_arr.append([int(id_list[row]), ans[row]])
    np_data = np.array(data_arr)

    # 写入文件
    pd_data = pd.DataFrame(np_data, columns=['id', 'y'])
    # print(pd_data)
    pd_data.to_csv('submit.csv', index=None)

    # 显示重要特征
    # plot_importance(model)
    # plt.show()
