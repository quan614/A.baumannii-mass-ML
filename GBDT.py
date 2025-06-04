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
import numpy as np
import pandas as pd
# GBoost

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

X = pd.read_csv('data.csv')
data = X.drop(['class'], axis=1)  # 数据
target = X.loc[:, 'class'].values  # 标签

print('****GBoost****')
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

n_estimators = [120, 130, 140, 145, 150, 160, 170, 175, 180]
learning_r = [0.1, 1, 0.01, 0.001]
parameters = {'n_estimators': n_estimators,
              'learning_rate': learning_r
              }
grid = GridSearchCV(GradientBoostingClassifier(),
                    param_grid=parameters,
                    cv=5,
                    n_jobs=-1)
grid.fit(X_train, y_train)

print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)

with open('paramater/GBoost.txt', 'w') as f:
    f.write("Best parameters: " + str(grid.best_params_) + '\n')
    f.write('Acc: %f' % grid.best_score_ + '\n\n')

    for p, s in zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score']):
        f.write(str(p) + ': ' + str(s) + '\n')
