from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd

# 加载数据
X = pd.read_csv('data.csv')
data = X.drop(['class'], axis=1)  # 数据
target = X.loc[:, 'class'].values  # 标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

print('****CatBoostClassifier****')

# 设置CatBoost的网格搜索参数
parameter = {
    'iterations': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'depth': [4, 6, 8, 10]
}

# 创建网格搜索对象
grid = GridSearchCV(estimator=CatBoostClassifier(verbose=0),  # verbose=0 关闭CatBoost训练过程中的冗长输出
                    param_grid=parameter,
                    cv=5)

# 开始训练
grid.fit(X_train, y_train)

# 输出最佳参数和得分
print('best_params_:', grid.best_params_)
print('best_score_:', grid.best_score_)

# 输出所有参数组合的交叉验证结果
for p, s in zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score']):
    print(p, s)

# 将结果保存到文件
with open('paramater/CatBoost.txt', 'w') as f:
    f.write("Best parameters: " + str(grid.best_params_) + '\n')
    f.write('Acc: %f' % grid.best_score_ + '\n\n')

    for p, s in zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score']):
        f.write(str(p) + ': ' + str(s) + '\n')
