"""
@filename: lasso_ft_sel.py
@author: James Rolfe
@date: 20170324
@reqs: trainTargets.csv, trainTargets.csv, testPredictors.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from math import sqrt, floor, ceil

## read all the data into a pd.DataFrame
# a sample of how the resulting data should be submitted

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

# remove colinear features as detected by collinear experiment
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

def get_opt_alph():
    ## define alphas to test over, must be len() = 10
    # alphas = [0.01, 0.1, 1, 10, 50, 100, 200, 300, 500, 1000]
    # ^ opt alpha was 1
    # alphas = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 5]
    # ^ opt alpha was 1
    alphas = [0.95, 0.98, 0.99, 1, 1.01, 1.05, 1.1, 1.15, 1.175 , 1.19]
    # ^ opt is still 1? -- 20170319
    # list of errors associated with each alpha
    errors = [1] * 10

    # list of where to split data
    s = [df_Y.shape[0]/10] * 9
    s = [a*b for a,b in zip(s, range(1, 10))]

    for a in alphas:
        i = alphas.index(a)

        # create linear regression object
        regr = linear_model.Lasso(alpha=a, max_iter=3000)
        # train on splits 0 to 8/9
        regr.fit(df_X[:s[8]], df_Y[:s[8]])
        # predict on splits 8/9 to 9/9
        preds = regr.predict(df_X[s[8]:])
        constrain_preds(preds)
        # save error
        errors[i] += mean_squared_error(df_Y[s[8]:], preds)

        # create linear regression object
        regr = linear_model.Lasso(alpha=a, max_iter=3000)
        # train on splits 1/9 to 9/9
        regr.fit(df_X[s[0]:], df_Y[s[0]:])
        # predict on splits 0/9 to 1/9
        preds = regr.predict(df_X[:s[0]])
        constrain_preds(preds)
        # save error
        errors[i] += mean_squared_error(df_Y[:s[0]], preds)

        # create linear regression object
        regr = linear_model.Lasso(alpha=a, max_iter=3000)
        # train on splits 2/9 to 1/9
        regr.fit(pd.concat([df_X[s[1]:], df_X[:s[0]]], axis=0), pd.concat([df_Y[s[1]:], df_Y[:s[0]]], axis=0))
        # predict on splits 1/9 to 2/9
        preds = regr.predict(df_X[s[0]:s[1]])
        constrain_preds(preds)
        # save error
        errors[i] += mean_squared_error(df_Y[s[0]:s[1]], preds)

        # create linear regression object
        regr = linear_model.Lasso(alpha=a, max_iter=3000)
        # train on splits 3/9 to 2/9
        regr.fit(pd.concat([df_X[s[2]:], df_X[:s[1]]], axis=0), pd.concat([df_Y[s[2]:], df_Y[:s[1]]], axis=0))
        # predict on splits 2/9 to 3/9
        preds = regr.predict(df_X[s[1]:s[2]])
        constrain_preds(preds)
        # save error
        errors[i] += mean_squared_error(df_Y[s[1]:s[2]], preds)

        # create linear regression object
        regr = linear_model.Lasso(alpha=a, max_iter=3000)
        # train on splits 4/9 to 3/9
        regr.fit(pd.concat([df_X[s[3]:], df_X[:s[2]]], axis=0), pd.concat([df_Y[s[3]:], df_Y[:s[2]]], axis=0))
        # predict on splits 3/9 to 4/9
        preds = regr.predict(df_X[s[2]:s[3]])
        constrain_preds(preds)
        # save error
        errors[i] += mean_squared_error(df_Y[s[2]:s[3]], preds)

        # create linear regression object
        regr = linear_model.Lasso(alpha=a, max_iter=3000)
        # train on splits 5/9 to 4/9
        regr.fit(pd.concat([df_X[s[4]:], df_X[:s[3]]], axis=0), pd.concat([df_Y[s[4]:], df_Y[:s[3]]], axis=0))
        # predict on splits 4/9 to 5/9
        preds = regr.predict(df_X[s[3]:s[4]])
        constrain_preds(preds)
        # save error
        errors[i] += mean_squared_error(df_Y[s[3]:s[4]], preds)

        # create linear regression object
        regr = linear_model.Lasso(alpha=a, max_iter=3000)
        # train on splits 6/9 to 5/9
        regr.fit(pd.concat([df_X[s[5]:], df_X[:s[4]]], axis=0), pd.concat([df_Y[s[5]:], df_Y[:s[4]]], axis=0))
        # predict on splits 4/9 to 5/9
        preds = regr.predict(df_X[s[4]:s[5]])
        constrain_preds(preds)
        # save error
        errors[i] += mean_squared_error(df_Y[s[4]:s[5]], preds)

        # create linear regression object
        regr = linear_model.Lasso(alpha=a, max_iter=3000)
        # train on splits 7/9 to 6/9
        regr.fit(pd.concat([df_X[s[6]:], df_X[:s[5]]], axis=0), pd.concat([df_Y[s[6]:], df_Y[:s[5]]], axis=0))
        # predict on splits 6/9 to 7/9
        preds = regr.predict(df_X[s[5]:s[6]])
        constrain_preds(preds)
        # save error
        errors[i] += mean_squared_error(df_Y[s[5]:s[6]], preds)

        # create linear regression object
        regr = linear_model.Lasso(alpha=a, max_iter=3000)
        # train on splits 8/9 to 7/9
        regr.fit(pd.concat([df_X[s[7]:], df_X[:s[6]]], axis=0), pd.concat([df_Y[s[7]:], df_Y[:s[6]]], axis=0))
        # predict on splits 6/9 to 7/9
        preds = regr.predict(df_X[s[6]:s[7]])
        constrain_preds(preds)
        # save error
        errors[i] += mean_squared_error(df_Y[s[6]:s[7]], preds)

        # create linear regression object
        regr = linear_model.Lasso(alpha=a, max_iter=3000)
        # train on splits 9/9 to 8/9
        regr.fit(pd.concat([df_X[s[8]:], df_X[:s[7]]], axis=0), pd.concat([df_Y[s[8]:], df_Y[:s[7]]], axis=0))
        # predict on splits 8/9 to 9/9
        preds = regr.predict(df_X[s[7]:s[8]])
        constrain_preds(preds)
        # save error
        errors[i] += mean_squared_error(df_Y[s[7]:s[8]], preds)

    errors = [x/10 for x in errors]
    return errors, alphas


# find training constraints
lat_max = max(lats)
lat_min = min(lats)
long_max = max(longs)
long_min = min(longs)
def constrain_preds(predictions):
    # enforce training constraints on the longitudes and latitudes
    for row in range(predictions.shape[0]):
        # round the latitudes
        if predictions[row][0] > 0:
            predictions[row][0] = ceil(predictions[row][0])
        else:
            predictions[row][0] = floor(predictions[row][0])

        # round the longitudes
        if predictions[row][1] > 0:
            predictions[row][1] = ceil(predictions[row][1])
        else:
            predictions[row][1] = floor(predictions[row][1])

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

def get_opt_num_feats(df_X, df_Y, model):
    n_feats = len(df_X.columns)
    errors = [0]*n_feats
    track_feats = range(1, n_feats+1)

    # list of numbers of features to try
    iters = range(1, n_feats+1)

    # list of where to split data
    s = [df_Y.shape[0]/10] * 9
    s = [a*b for a,b in zip(s, range(1, 10))]
    i = 0

    # get optimal alpha
    opt_alph = get_opt_alph()

    # create linear regression object
    regr = linear_model.Lasso(alpha=opt_alph, max_iter=3000)

    # this gets the best amount of features for models with n_features=k-1:1
    for n in iters:
        sel = RFE(model, n, step=1)
        mat_X = sel.fit_transform(df_X.values, df_Y)

        # train on splits 0 to 8/9
        regr.fit(mat_X[:s[8]], df_Y[:s[8]])
        # predict on splits 8/9 to 9/9
        preds = regr.predict(mat_X[s[8]:])
        # save error
        errors[i] += mean_squared_error(df_Y[s[8]:], preds)

        # train on splits 1/9 to 9/9
        regr.fit(mat_X[s[0]:], df_Y[s[0]:])
        # predict on splits 0/9 to 1/9
        preds = regr.predict(mat_X[:s[0]])
        # save error
        errors[i] += mean_squared_error(df_Y[:s[0]], preds)

        # train on splits 2/9 to 1/9
        regr.fit(np.concatenate([mat_X[s[1]:], mat_X[:s[0]]], axis=0), np.concatenate([df_Y[s[1]:], df_Y[:s[0]]], axis=0))
        # predict on splits 1/9 to 2/9
        preds = regr.predict(mat_X[s[0]:s[1]])
        # save error
        errors[i] += mean_squared_error(df_Y[s[0]:s[1]], preds)

        # train on splits 3/9 to 2/9
        regr.fit(np.concatenate([mat_X[s[2]:], mat_X[:s[1]]], axis=0), np.concatenate([df_Y[s[2]:], df_Y[:s[1]]], axis=0))
        # predict on splits 2/9 to 3/9
        preds = regr.predict(mat_X[s[1]:s[2]])
        # save error
        errors[i] += mean_squared_error(df_Y[s[1]:s[2]], preds)

        # train on splits 4/9 to 3/9
        regr.fit(np.concatenate([mat_X[s[3]:], mat_X[:s[2]]], axis=0), np.concatenate([df_Y[s[3]:], df_Y[:s[2]]], axis=0))
        # predict on splits 3/9 to 4/9
        preds = regr.predict(mat_X[s[2]:s[3]])
        # save error
        errors[i] += mean_squared_error(df_Y[s[2]:s[3]], preds)

        # train on splits 5/9 to 4/9
        regr.fit(np.concatenate([mat_X[s[4]:], mat_X[:s[3]]], axis=0), np.concatenate([df_Y[s[4]:], df_Y[:s[3]]], axis=0))
        # predict on splits 4/9 to 5/9
        preds = regr.predict(mat_X[s[3]:s[4]])
        # save error
        errors[i] += mean_squared_error(df_Y[s[3]:s[4]], preds)

        # train on splits 6/9 to 5/9
        regr.fit(np.concatenate([mat_X[s[5]:], mat_X[:s[4]]], axis=0), np.concatenate([df_Y[s[5]:], df_Y[:s[4]]], axis=0))
        # predict on splits 4/9 to 5/9
        preds = regr.predict(mat_X[s[4]:s[5]])
        # save error
        errors[i] += mean_squared_error(df_Y[s[4]:s[5]], preds)

        # train on splits 7/9 to 6/9
        regr.fit(np.concatenate([mat_X[s[6]:], mat_X[:s[5]]], axis=0), np.concatenate([df_Y[s[6]:], df_Y[:s[5]]], axis=0))
        # predict on splits 6/9 to 7/9
        preds = regr.predict(mat_X[s[5]:s[6]])
        # save error
        errors[i] += mean_squared_error(df_Y[s[5]:s[6]], preds)

        # train on splits 8/9 to 7/9
        regr.fit(np.concatenate([mat_X[s[7]:], mat_X[:s[6]]], axis=0), np.concatenate([df_Y[s[7]:], df_Y[:s[6]]], axis=0))
        # predict on splits 6/9 to 7/9
        preds = regr.predict(mat_X[s[6]:s[7]])
        # save error
        errors[i] += mean_squared_error(df_Y[s[6]:s[7]], preds)

        # train on splits 9/9 to 8/9
        regr.fit(np.concatenate([mat_X[s[8]:], mat_X[:s[7]]], axis=0), np.concatenate([df_Y[s[8]:], df_Y[:s[7]]], axis=0))
        # predict on splits 8/9 to 9/9
        preds = regr.predict(mat_X[s[7]:s[8]])
        # save error
        errors[i] += mean_squared_error(df_Y[s[7]:s[8]], preds)

        # iterate error list index
        i += 1

    # normalize all the errors given 10-fold CV
    errors = [x/10 for x in errors]

    # finc the num_features that the model with the lowest error has
    idx = errors.index(min(errors))
    return track_feats[idx], opt_alph

# transform a dataframe to only include indexed features with '1' relevance in
# the ranking vector
def transform_test_X(ranking, test_df):
    # only record the features with relevance ranking of '1'
    ranking = np.where(ranking == 1)

    # flatten form 2D to 1D
    ranking = list(ranking[0].flatten())
    # init DataFrame to be returned
    ret_df = pd.DataFrame()

    # re-add each feature and its corresponding column name
    for x in ranking:
        c_name = 'feature' + str(x+1)
        ret_df.insert(loc=len(ret_df.columns), column=c_name,
                value=test_df.iloc[:, x].values)

    return ret_df

opt_num_lat_feats, opt_alph = get_opt_num_feats(df_X, lats, regr)
print('optimal number of features for lat: ' + str(opt_num_lat_feats))

opt_num_long_feats, opt_alph = get_opt_num_feats(df_X, longs, regr)
print('optimal number of features for long: ' + str(opt_num_long_feats))

# recreate linear regression object
regr = linear_model.Lasso(alpha=opt_alph, max_iter=3000)

# rebuild the optimal model
sel = RFE(regr, opt_num_lat_feats, step=1)
sel.fit(df_X.values, lats)
mat_X = sel.transform(df_X.values)

# transform the test predictors to only include the selected features
df_test_ft_sel = transform_test_X(sel.ranking_, df_test_X)

# print(df_test_ft_sel)

# fit (lasso) the transformed train predictors on the train features
regr.fit(mat_X, lats)
# predict using the transformed test predictors
lat_preds = regr.predict(df_test_ft_sel.values)
# lat_error = sqrt(mean_squared_error(lats, lat_preds))
# print('lat error: ' + str(lat_error))

sel = RFE(regr, opt_num_long_feats, step=1)
sel.fit(df_X.values, longs)
mat_X = sel.transform(df_X.values)

# transform the test predictors to only include the selected features
df_test_ft_sel = transform_test_X(sel.ranking_, df_test_X)

# print(df_test_ft_sel)

# fit (OLS) the transformed train predictors on the train features
regr.fit(mat_X, longs)
# predict using the transformed test predictors
long_preds = regr.predict(df_test_ft_sel.values)
# long_error = sqrt(mean_squared_error(longs, long_preds))
# print('long error: ' + str(long_error))

# put the lat and long predictions together
preds = np.column_stack((lat_preds, long_preds))

# transform outputs
constrain_preds(preds)

# convert back to a dataframe
df_pred = pd.DataFrame(preds, columns=['lat','long'])

# print(sqrt(mean_squared_error(df_Y, df_pred)))

# add the original index column back
df_pred.insert(loc=0, column='index', value=test_X_index.values)

# print(df_pred)

# export to csv for submission
df_pred.to_csv('lasso_ft_sel.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
