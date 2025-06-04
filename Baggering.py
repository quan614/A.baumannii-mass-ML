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
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd

X = pd.read_csv('data.csv')
data = X.drop(['class'], axis=1)  # 数据
target = X.loc[:, 'class'].values  # 标签

print('****Bagging****')
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

n_estimators = [10, 30, 50, 70, 80, 120, 130, 140, 145, 150, 160, 170, 175, 180, 185]
max_features = range(1, 20)
parameters = {'n_estimators': n_estimators,
              'max_features': max_features}
grid = GridSearchCV(BaggingClassifier(),
                    param_grid=parameters,
                    cv=5,
                    n_jobs=-1)
grid.fit(X_train, y_train)

print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)

with open('paramater/Bagging.txt', 'w') as f:
    f.write("Best parameters: " + str(grid.best_params_) + '\n')
    f.write('Acc: %f' % grid.best_score_ + '\n\n')

    for p, s in zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score']):
        f.write(str(p) + ': ' + str(s) + '\n')
