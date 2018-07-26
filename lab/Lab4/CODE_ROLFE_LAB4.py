"""
@filename: lms.py
@author: James Rolfe
@date: 20170327
@reqs: LMSalgtrain.csv, LMSalgtest.csv 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

TRAIN_CSV = 'LMSalgtrain.csv' 
TEST_CSV = 'LMSalgtest.csv'

def read_data(filename):
    # use pandas dataframe to read values from csv
    df = pd.read_csv(filename)

    # separate inputs and outputs
    X = df.drop(['z', 'estimate'], axis=1)
    # add bias term to inputs
    X.insert(len(X.columns), 'bias', np.ones((X.shape[0], 1)),
            allow_duplicates=True)

    Y = df['estimate']
    Z = df['z']
    return X, Y, Z

'''
@params: z: np.matrix(int|float)
@post: call to np.exp()
@returns: np.matrix(int|float)
'''
def sigmoid(z):
    # use the definition of the sigmoid function
    return 1.0 / (1.0 + np.exp(-1.0 * z))

def ols(X_train, Y_train, X_test):
    # create linear regression object
    regr = linear_model.LinearRegression()

    # train the model without the index column
    regr.fit(X_train, Y_train)

    # get predictions from trained model
    predictions = regr.predict(X_test)

    return predictions, regr.coef_

def mse(Y_test, preds):
    mse = mean_squared_error(Y_test, preds)
    return mse

def update_weights(X, ts, W, eta):
    for idx in range(X.shape[0]):
        y = W.transpose().dot(X[idx][:].transpose())
        err = ts[idx] - y
        delta_W = np.matrix(eta * err * X[idx][:]).transpose()
        W += delta_W
    return W

def delta(X, ts, eta, iters):
    W = np.zeros((X.shape[1], 1))
    while(True):
        W = update_weights(X, ts, W, eta)
        ys = W.transpose().dot(X.transpose())
        MSE = mse(list(ts), ys.tolist()[0])
        iters -= 1
        if iters < 1:
            return MSE, W

'''
This function takes a prediction array full of class values and a ground truth
data frame full of class values and returns a confusion matrix data frame.
@params: prediction:array, ground_truth:pandas.DataFrame
@post: prints the conf matrix to stdout
@returns: confMatrix:pandas.DataFrame
'''
def confusionMatrix(df_Y, prediction, ground_truth):
    # get the column names for the conf matrix
    names = list(df_Y['z'].unique())
    # names.reverse() # make the list so the conf matrix comes out as requested
    
    # initialize the data that will make up the conf matrix data frame
    conf_data = [[0 for i in range(len(names))] for j in range(len(names))]
    confMatrix = pd.DataFrame(conf_data, index=names, columns=names)

    # names the columns and indices of the conf matrix
    confMatrix.index.names = ['actual']
    confMatrix.columns.names = ['predicted']
    
    # load the data
    for i in range(len(prediction)):
        confMatrix.loc[ground_truth[i], prediction[i]] += 1

    print('\nconfusion matrix')
    print(confMatrix)

    return confMatrix

'''
This function calculates the trace of a pandas data frame. The dataframe is
assumed to be square.
@params: df:pandas.DataFrame
@post: none
@returns: float
'''
def df_trace(df):
    # sum the left-to-right diagnal of the data frame 
    return sum(df.loc[i,i] for i in range(len(df)))

'''
conf = confusionMatrix(prediction, groundtruth)

# calculate the accuracy as the number of true positives plus true negatives
# over the total classifications. multiplied by 100 for percentage
accuracy = (float(df_trace(conf))/conf.sum().sum())*100

print('\naccuracy = ' + str(accuracy) + '%')
'''

# get data frames of data from csvs
X_train, Y_train, Z_train = read_data(TRAIN_CSV)
X_test, Y_test, Z_test = read_data(TEST_CSV)

# get the OLS solution
ols_preds, ols_coefs = ols(X_train, Y_train, X_test)

# get lms solution
lms_train_mse, lms_coefs = delta(X_train.values, Y_train.values, 0.0001, 1000)

print('weights for LMS:')
print(lms_coefs)

print('training mse for LMS (full batch)[eta=0.0001; iters=1000]: %.3f'% lms_train_mse)
lms_preds = X_test.values.dot(lms_coefs)
print('testing mse for LMS (full batch)[eta=0.0001; iters=1000]: %.3f'% mse(Y_test, lms_preds))

print('weights for OLS:')
print(ols_coefs)

print('training mse for OLS: %.3f'% mse(Y_train, ols(X_train, Y_train, X_train)[0]))
print('testing mse for OLS: %.3f'% mse(Y_test, ols_preds))
