"""
@filename: lda.py
@author: James Rolfe
@date: 20170312
@reqs: trainTargets.csv, trainTargets.csv, testPredictors.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_squared_error
from math import sqrt

# training data sets
df_X = pd.read_csv('trainPredictors.csv')
df_Y = pd.read_csv('trainTargets.csv')
# save index columns for X and Y
X_index = df_X['index']
Y_index = df_Y['index']
# remove index columns, since they don't help with model fit
df_X = df_X.drop('index', axis=1)
df_Y = df_Y.drop('index', axis=1)

lats = df_Y['lat'].values
longs = df_Y['long'].values

# testing data set, this is what the submission should be based upon
df_test_X = pd.read_csv('testPredictors.csv')
test_X_index = df_test_X['index']
df_test_X = df_test_X.drop('index', axis=1)

# remove colinear features as detected by stata experiment
cols1 = range(18, 29)
cols2 = range(47, 58)
cols3 = range(76, 87)
cols4 = range(105, 116)
df_X.drop(df_X.columns[cols4], axis=1, inplace=True)
df_X.drop(df_X.columns[cols3], axis=1, inplace=True)
df_X.drop(df_X.columns[cols2], axis=1, inplace=True)
df_X.drop(df_X.columns[cols1], axis=1, inplace=True)
df_test_X.drop(df_test_X.columns[cols4], axis=1, inplace=True)
df_test_X.drop(df_test_X.columns[cols3], axis=1, inplace=True)
df_test_X.drop(df_test_X.columns[cols2], axis=1, inplace=True)
df_test_X.drop(df_test_X.columns[cols1], axis=1, inplace=True)

# create linear regression object
regr = LinearDiscriminantAnalysis()

# train the model without the index column
regr.fit(df_X.values, list(lats))
# make predictions without the index column
lat_preds = regr.predict(df_test_X)

# train the model without the index column
regr.fit(df_X.values, list(longs))
# make predictions without the index column
long_preds = regr.predict(df_test_X)

# put the lat and long predictions together
predictions = np.column_stack((lat_preds, long_preds))

# find training constraints
lat_max = max(lats)
lat_min = min(lats)
long_max = max(longs)
long_min = min(longs)
# enforce training constraints on the longitudes and latitudes
for row in range(predictions.shape[1]):
    # set a constraint on the latitudes
    if predictions[0][row] > lat_max:
        predictions[0][row] = lat_max
    if predictions[0][row] < lat_min:
        predictions[0][row] = lat_min

    # set a constraint on the longitudes
    if predictions[1][row] > long_max:
        predictions[1][row] = long_max
    if predictions[1][row] < long_min:
        predictions[1][row] = long_min

df_pred = pd.DataFrame(predictions, columns=['lat','long'])

# print('mean error: %.3f' % sqrt(mean_squared_error(df_Y, df_pred)))

df_pred.insert(loc=0, column='index', value=test_X_index.values)

# export to csv for submission
df_pred.to_csv('lda.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
