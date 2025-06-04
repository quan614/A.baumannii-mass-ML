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
#from xgboost import XGBClassifier
import numpy as np
import pandas as pd

X = pd.read_csv('data.csv')
data = X.drop(['class'], axis=1)  # 数据
target = X.loc[:, 'class'].values  # 标签

print('****RF****')
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

n_estimators = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
# n_estimators = [100, 110, 120, 130, 140, 150, 160, 170, 180]
max_depth = range(1, 30)
# max_depth = range(25, 30)
# criterions = ['gini', 'entropy']
criterions = ['gini', 'entropy']

parameters = {'n_estimators': n_estimators,
              'max_depth': max_depth,
              'criterion': criterions
              }

grid = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)

with open('paramater/RF.txt', 'w') as f:
    f.write("Best parameters: " + str(grid.best_params_) + '\n')
    f.write('Acc: %f' % grid.best_score_ + '\n\n')

    for p, s in zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score']):
        f.write(str(p) + ': ' + str(s) + '\n')