"""
@filename: LinearRegression.py
@author: James Rolfe
@date: 20170302
@reqs: HousingData_LinearRegression.csv 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_CSV = 'HousingData_LinearRegression.csv'

# read all the data into a pd.DataFrame
df = pd.read_csv(DATA_CSV)

# normalize ALL the data by feature
for feat_name in df.columns:
    mean = df[feat_name].mean()
    std = df[feat_name].std()
    df[feat_name] = (df[feat_name] - mean) / std

# Pre-process the data
X = df
Y = df.copy(True)

# X only contains predictors
X = X.drop('Price(USD)', axis=1)
# Add intercept column of all 1s
X['Intercept'] = pd.Series(np.ones(len(X.index)))

# Y is only the response variable
Y = Y['Price(USD)']

# convert them all to matrices in order to make computations easier
X = np.matrix(X.values)  
Y = np.matrix(Y.values)

# transpose Y to match the dimensions of X
Y = Y.transpose()

print(X)
print(Y)

# initialize beta coefficients (theta) to 0s of a length equal to the number of
# columns in X
beta = np.zeros((X.shape[1],1)) 

'''
@params: beta: np.matrix(float)
@post: use of global variables X and Y
@returns: float
'''
def costFunction(beta):
    '''
    Compute the Least Square Cost Function.
    Return the calculated cost function.
    '''
    res = 1.0 / (2 * X.shape[0]) * (X * beta - Y).transpose() * (X * beta - Y)
    return res.item((0,0))

'''
@params: beta: np.matrix(float)
@post: use of global variables X and Y
@returns: np.matrix(float)
'''
def gradientDescentHelper(beta, alpha):
    # calculate gradient by deriving the cost function
    grad = X.transpose() * (X * beta - Y)
    
    # calculate new beta coeffecients using gradient and learning rate (alpha)
    beta = beta - ((alpha / X.shape[0]) * grad)
    
    return beta

'''
@params: beta: np.matrix(float); alpha: float; iters: float
@post: calls to gradientDescentHelper() and costFunction()
@returns: np.matrix(float); list(float)
'''
def gradientDescent(beta, alpha, iters):
    # initialize list to save all the costs
    costs = []
    
    # iterate iters times, update the betas while descending towards the minima
    for i in range(iters):
        costs.append(costFunction(beta))
        beta = gradientDescentHelper(beta, alpha)
    
    return beta, costs

'''
@params: costs: list(float); title: string
@post: calls to plt functions; plot to matplotlib out
@returns: none
'''
def graphCost(costs, title):
    plt.scatter(range(len(costs)), costs)
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()

'''
@params: beta: np.matrix(float)
@post: use of global variables X and Y
@returns: np.matrix(float)
'''
def gradientDescentRidgeHelper(beta, alpha, ridgeLambda):
    # calculate gradient by deriving the cost function
    grad = X.transpose() *  (X * beta - Y)
    
    # calculate the L2 penalty by taking the derivitive of L2
    penalty = 2 * ridgeLambda * beta
    
    # calculate new beta coeffecients using gradient and learning rate (alpha)
    # and the L2 penalty
    beta = beta - (alpha * ((1.0 / X.shape[0]) * grad + penalty))
    
    return beta

'''
@params: beta: np.matrix(float); alpha: float; iters: float; ridgeLambda: float
@post: calls to gradientDescentHelper() and costFunction()
@returns: np.matrix(float); list(float)
'''
def gradientDescentRidge(beta, alpha, itersreg, ridgeLambda):
    # initialize list to save all the costs
    costs = []
    
    # iterate iters times, update the betas while descending towards the minima
    for i in range(itersreg):
        costs.append(costFunction(beta))
        beta = gradientDescentRidgeHelper(beta, alpha, ridgeLambda)
    
    return beta, costs

'''
@params: beta: np.matrix(float); reg: bool
@post: calls to gradientDescentRidge(), gradientDescent(), and graphCost()
@returns: float
'''
def MSE(beta, reg):
    # define the learning rate, iterations, and shrinkage parameters
    alpha = 0.01
    iters = 1500
    lam = 0.05
    
    # initialze costs list to be used later
    costs = []
    
    # initialize MSE to be returned later
    mse = 0

    if reg:
        beta, costs = gradientDescentRidge(beta, alpha, iters, lam)
        print('Betas with Regularization')
        print(beta.transpose())
        graphCost(costs, 'Error vs Training with Regularization')

        # the MSE will be 2 times the lowest cost
        mse = 2 * costs[-1]
    else:
        beta, costs = gradientDescent(beta, alpha, iters)
        print('\nBetas without Regularization')
        print(beta.transpose())
        graphCost(costs, 'Error vs Training without Regularization')

        # the MSE will be 2 times the lowest cost
        mse = 2 * costs[-1]
    return mse

# MSE for beta without regularization
print('MSE for Betas without Regularization: ' + str(MSE(beta, False)))

# MSE for beta with regularization
print('MSE for Betas with Regularization: ' + str(MSE(beta, True)))

# TODO: DELETE BELOW
#The final result wanted for this portion of the lab is the last plot defined earlier, the explanation regarding the 
#coeffecients of the parameters with Ridge Regression regularization, and the MSE. Please only include the
#MSE in the same .txt file as logistic regression results. Also let the print regResult[0] be there, but do NOT include the 
#outcome for this print in the .txt file. Add the final plot to your report and explain your algorithm, the plot, and 
#the MSE. Generally, what you did in this portion of the lab. Finally, explain the coeffiencts with regularization in your report -PDF file-. 
