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

print('****SVM****')
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

parameters = [
     {"kernel": ["rbf"],  "C": [0.0001, 0.001, 0.01, 0.1, 1], "gamma": [0.0001, 0.001, 0.01, 0.1, 1]},
     #{"kernel": ["poly"], "C": [0.0001, 0.001, 0.01, 0.1, 1], "degree": [1, 3, 5, 7, 9, 11]},
     #{"kernel": ["linear"], "C": [0.0001, 0.001, 0.01,0.1,1], "gamma": [0.0001, 0.001,0.01,0.1,1]},
     #{"kernel": ["sigmoid"], "C": [0.0001, 0.001, 0.01, 0.1, 1], "gamma": [1, 2, 3, 4]}
]
# C,gamma,kernel
svc = SVC(probability=True, decision_function_shape='ovo')  # 改为多分类器，设置参数decision_function = one against one
grid = GridSearchCV(svc, parameters, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
print("best parameters:", grid.best_params_)
print('训练集Acc : %f' % grid.best_score_)

for p, s in zip(grid.cv_results_['params'],
    grid.cv_results_['mean_test_score']):
    print(p, s)

with open('paramater/SVM.txt', 'w') as f:
    f.write("Best parameters: " + str(grid.best_params_) + '\n')
    f.write('Acc: %f' % grid.best_score_ + '\n\n')

    for p, s in zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score']):
        f.write(str(p) + ': ' + str(s) + '\n')