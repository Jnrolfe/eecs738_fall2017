"""
@filename: knn.py
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

from sklearn.neighbors import KNeighborsClassifier

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
k = model_selection.KFold(n_splits = 10)

ns = [1] # best is 90
results = [0] * len(ns)
i = 0

for n in ns:
    clf = KNeighborsClassifier(n_neighbors=n)
    result = model_selection.cross_val_score(clf, df_X.values,
        np.ravel(df_Y.values), cv=k)
    results[i] = result.mean()
    print(result.mean())
    i += 1
print('best n: ' + str(ns[results.index(max(results))]))

# predictions = clf.predict(df_test_X.values[:20000])
# print(predictions)
