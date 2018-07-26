"""
@filename: ols.py
@author: James Rolfe
@date: 20170324
@reqs: trainTargets.csv, trainTargets.csv, testPredictors.csvs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn import linear_model
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

# create linear regression object
regr = linear_model.LinearRegression()

# train the model without the index column
regr.fit(df_X, df_Y)

# make predictions without the index column
predictions = regr.predict(df_X)

# find training constraints
lat_max = max(lats)
lat_min = min(lats)
long_max = max(longs)
long_min = min(longs)
def constrain_preds(predictions):
    # enforce training constraints on the longitudes and latitudes
    for row in range(predictions.shape[0]):
        '''
        if predictions[row][0] > 0:
            predictions[row][0] = ceil(predictions[row][0])
        else:
            predictions[row][0] = floor(predictions[row][0])

        if predictions[row][1] > 0:
            predictions[row][1] = ceil(predictions[row][1])
        else:
            predictions[row][1] = floor(predictions[row][1])
        '''

        # set a constraint on the latitudes
        if predictions[row][0] > lat_max:
            predictions[row][0] = lat_max
        if predictions[row][0] < lat_min:
            predictions[row][0] = lat_min

        # set a constraint on the longitudes
        if predictions[row][1] > long_max:
            predictions[row][1] = long_max
        if predictions[row][1] < long_min:
            predictions[row][1] = long_min

df_pred = pd.DataFrame(predictions, columns=['lat','long'])

plt.plot(df_pred['lat'], label='predicted')
plt.plot(df_Y['lat'], label='actual')
plt.legend()
plt.title('Latitude Predicted vs Actual')
plt.show()

plt.plot(df_pred['long'], label='predicted')
plt.plot(df_Y['long'], label='actual')
plt.legend()
plt.title('Longitude Predicted vs Actual')
plt.show()

# print('mean error: %.3f' % sqrt(mean_squared_error(df_Y, df_pred)))

df_pred.insert(loc=0, column='index', value=X_index.values)

# export to csv for submission
# df_pred.to_csv('ols.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
