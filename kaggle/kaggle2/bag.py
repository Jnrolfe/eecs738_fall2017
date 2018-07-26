"""
@filename: bag.py
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

# get models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
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
# k = model_selection.KFold(n_splits=10)
# add each model to the ensemble
b_est = KNeighborsClassifier(n_neighbors=1)

# create voting ensemble
bag_model = BaggingClassifier(base_estimator=b_est, n_estimators=75)

# get cv result
# result = model_selection.cross_val_score(bag_model, df_X.values,
#    np.ravel(df_Y.values), cv=k)
# print(result.mean())

bag_model.fit(df_X.values, np.ravel(df_Y.values))
preds = bag_model.predict(df_test_X.values)

df_pred = pd.DataFrame(preds, columns=['coverType_1to7'])
df_pred.insert(loc=0, column='id', value=np.ravel(df_test_X_ids.values))
print(df_pred[:10])
df_pred.to_csv('bag_knn_75.csv', index=False)

# predictions = clf.predict(df_test_X.values[:20000])
# print(predictions)
