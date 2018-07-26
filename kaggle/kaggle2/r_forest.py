"""
@filename: r_forest.py
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

from sklearn.metrics import accuracy_score, confusion_matrix

# get models
from sklearn.ensemble import RandomForestClassifier
# ============================================================================
## preprocess data
# training data sets
df_X = pd.read_csv('trainPredictors.csv')
df_Y = pd.read_csv('trainTargets.csv')

X_ids = df_X['id']
Y_ids = df_Y['id']

df_X = df_X.drop('id', axis=1)
df_Y = df_Y.drop('id', axis=1)

# testing data set, this is what the submission should be based upon
df_test_X = pd.read_csv('testPredictors.csv')
df_test_X_ids = df_test_X['id']
df_test_X = df_test_X.drop('id', axis=1)

# testing
# print(df_test_X.shape)
# print(np.ravel(df_Y.values[:10]))

# ============================================================================
train_X, test_X, train_Y, test_Y = model_selection.train_test_split(df_X, df_Y,
    test_size=.1)

clf = RandomForestClassifier(n_estimators=90)
clf.fit(train_X.values, np.ravel(train_Y.values))
preds = clf.predict(test_X)
print(confusion_matrix(test_Y, preds))
# print(accuracy_score(preds, np.ravel(test_Y.values)))

# df_pred = pd.DataFrame(preds, columns=['coverType_1to7'])
# df_pred.insert(loc=0, column='id', value=np.ravel(df_test_X_ids.values))
# print(df_pred[:10])
# df_pred.to_csv('r_forest_90.csv', index=False)

'''
# prep cv
k = model_selection.KFold(n_splits = 10)

ns = [90] # best is 90
results = [0] * len(ns)
i = 0

for n in ns:
    clf = RandomForestClassifier(n_estimators=n)
    result = model_selection.cross_val_score(clf, df_X.values,
        np.ravel(df_Y.values), cv=k)
    results[i] = result.mean()
    print(result.mean())
    i += 1
print('best n: ' + str(ns[results.index(max(results))]))
'''
# predictions = clf.predict(df_test_X.values[:20000])
# print(predictions)
