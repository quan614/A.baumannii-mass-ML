from scipy.linalg import svd
from scipy.sparse.linalg import eigen
from scipy.sparse.linalg._isolve import lsqr
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
from xgboost import XGBClassifier
import numpy as np
import pandas as pd

X = pd.read_csv('data.csv')
data = X.drop(['class'], axis=1)  # 数据
target = X.loc[:, 'class'].values  # 标签

print('****XGBClassifier****')
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# parameter = {'n_estimators':[50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180],'learning_rate':[0.1, 1, 0.01, 0.001]}
parameter = {'n_estimators':[50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180],'learning_rate':[0.1, 1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(estimator=XGBClassifier(), param_grid=parameter, cv=5)
grid.fit(X_train, y_train)
print('best_params_:', grid.best_params_)
print('best_score_:', grid.best_score_)
for p, s in zip(grid.cv_results_['params'],
    grid.cv_results_['mean_test_score']):
    print(p, s)

with open('paramater/XGB.txt', 'w') as f:
    f.write("Best parameters: " + str(grid.best_params_) + '\n')
    f.write('Acc: %f' % grid.best_score_ + '\n\n')

    for p, s in zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score']):
        f.write(str(p) + ': ' + str(s) + '\n')