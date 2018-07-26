"""
@filename: voting.py
@author: James Rolfe
@date: 20170507
@reqs: trainTargets.csv, trainTargets.csv, testPredictors.csvs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

# cross validation
from sklearn import model_selection

from sklearn.metrics import accuracy_score

# get models
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
# ============================================================================
## preprocess data
# training data sets
df_X = pd.read_csv('trainPredictors.csv')
df_Y = pd.read_csv('trainTargets.csv')

X_ids = df_X['id']
Y_ids = df_Y['id']

df_X = df_X.drop('id', axis=1)
df_Y = df_Y.drop('id', axis=1)

print(df_X.shape)
print(df_Y.shape)

# testing data set, this is what the submission should be based upon
df_test_X = pd.read_csv('testPredictors.csv')
df_test_X_ids = df_test_X['id']
df_test_X = df_test_X.drop('id', axis=1)

# testing
# print(df_test_X.shape)
# print(np.ravel(df_Y.values[:10]))

# ============================================================================

# prep cv
# k = model_selection.KFold(n_splits = 10)

train_X, test_X, train_Y, test_Y = model_selection.train_test_split(df_X, df_Y,
    test_size=.1)

# code below is based upon code from http://machinelearningmastery.com/ensemble-
# machine-learning-algorithms-python-scikit-learn/
# add each model to the ensemble
models = []
'''
m0 = RandomForestClassifier(n_estimators=50)
models.append(('r_forest', m0))
'''
m1 = RandomForestClassifier(n_estimators=60)
models.append(('r_forest', m1))
m2 = RandomForestClassifier(n_estimators=70)
models.append(('r_forest', m2))
m3 = RandomForestClassifier(n_estimators=80)
models.append(('r_forest', m3))
m4 = RandomForestClassifier(n_estimators=90)
models.append(('r_forest', m4))
m5 = KNeighborsClassifier(n_neighbors=1)
models.append(('knn', m5))
m6 = KNeighborsClassifier(n_neighbors=2)
models.append(('knn', m6))
m7 = KNeighborsClassifier(n_neighbors=3)
models.append(('knn', m7))
m8 = KNeighborsClassifier(n_neighbors=4)
models.append(('knn', m8))
m9 = KNeighborsClassifier(n_neighbors=5)
models.append(('knn', m9))
# create voting ensemble
e = VotingClassifier(models, weights=[0.8,0.9,1,1.1,1.1,1.1,1,0.9,0.8])
e.fit(train_X.values, np.ravel(train_Y.values))
preds = e.predict(test_X.values)
print(accuracy_score(np.ravel(test_Y.values), preds))
'''
df_pred = pd.DataFrame(preds, columns=['coverType_1to7'])
df_pred.insert(loc=0, column='id', value=np.ravel(df_test_X_ids.values))
print(df_pred[:10])
df_pred.to_csv('voting_4forst_5knn_weights.csv', index=False)
'''
# get cv result
# result = model_selection.cross_val_score(e, df_X.values, np.ravel(df_Y.values)
#    , cv=k)
# print(result.mean())

# predictions = clf.predict(df_test_X.values[:20000])
# print(predictions)
