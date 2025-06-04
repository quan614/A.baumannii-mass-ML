from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from xgboost import XGBClassifier
import numpy as np
import pandas as pd

X = pd.read_csv('data.csv')
data = X.drop(['class'], axis=1)  # 数据
target = X.loc[:, 'class'].values  # 标签

print('****决策树****')
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

tree_param_grid = {'criterion': ['entropy',"gini"], "max_depth" : range (1, 30), "max_features":[21, 22, 23, 24, 25, 26, 28, 29, 30, 'auto'],
                   'max_depth': range (1, 30)}
# min_samples_leaf, max_depth
grid = GridSearchCV(DecisionTreeClassifier(), param_grid=tree_param_grid,
                    cv=5)  # criterion，gini或entropy
grid.fit(X_train, y_train)
print('best_params_:', grid.best_params_)
print('best_score_:', grid.best_score_)

with open('paramater/DT.txt', 'w') as f:
    f.write("Best parameters: " + str(grid.best_params_) + '\n')
    f.write('Acc: %f' % grid.best_score_ + '\n\n')

    for p, s in zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score']):
        f.write(str(p) + ': ' + str(s) + '\n')