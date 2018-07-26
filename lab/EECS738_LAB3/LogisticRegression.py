"""
@filename: LogisticRegression.py
@author: James Rolfe
@date: 20170302
@reqs: LogisticData.csv 
"""

from __future__ import division
import numpy as np
import pandas as pd
import scipy.optimize as opt

DATA_CSV = 'LogisticData.csv' 

# read all the data into a pd.DataFrame
df = pd.read_csv(DATA_CSV)

# save ground truth for accuracy calculation later
groundTruth = df['Admitted']

# normalize data by feature
for feat_name in df.columns:
    if feat_name == 'Admitted':
        # do not normalize Y, since they're booleans
        continue
    mean = df[feat_name].mean()
    std = df[feat_name].std()
    df[feat_name] = (df[feat_name] - mean) / std

# Pre-process the data
X = df 
Y = df.copy(True)

# X only contains predictors
X = X.drop('Admitted', axis=1)
# Add intercept column of all 1s
X['Intercept'] = pd.Series(np.ones(len(X.index)))

# Y is only the response variable
Y = Y['Admitted']

# initialize beta coefficients (theta) to 0s of a length equal to the number of
# columns in X
beta = np.zeros((len(X.columns),1)) 

# convert them all to matrices in order to make computations easier
X = np.matrix(X.values)  
Y = np.matrix(Y.values)

# transpose Y to match the dimensions of X
Y = Y.transpose()

'''
@params: z: np.matrix(int|float)
@post: call to np.exp()
@returns: np.matrix(int|float)
'''
def sigmoid(z):
    # use the definition of the sigmoid function
    return 1.0 / (1.0 + np.exp(-1.0 * z))

'''
@params: beta: np.matrix(float)
@post: calls to sigmoid(), np.log(); use of global variables X and Y
@returns: float
'''
def costFunction(beta):
    # use the cost function provided where \hat(y) = sigmoid(X * beta)
    result = (-1.0 / X.shape[0]) * ((Y.transpose() * np.log(sigmoid(X*beta))) \
            + ((1 - Y).transpose() * np.log(1 - sigmoid(X*beta))))
    return result.item((0, 0))

'''
@params: beta: np.matrix(float); alpha: float
@post: calls to sigmoid(); use of global variables X and Y
@returns: np.matrix(float)
'''
def gradientDescentHelper(beta, alpha):
    # calculate gradient by deriving the cost function
    grad = X.transpose() * (sigmoid(X * beta) - Y)

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
        beta = gradientDescentHelper(beta, alpha)
        costs.append(costFunction(beta))
    
    return beta, costs

'''
@params: beta: np.array(float)
@post: calls to sigmoid(); use of global variables X and Y
@returns: the gradient function
'''
def gradientOpt(beta):
    # the scipy optimize function passes beta as a list, as a result it must
    # be re-cast as a np.matrix in order for this algorithm to work
    beta = np.matrix(beta).transpose()
    
    # calculate gradient by deriving the cost function
    grad = X.transpose() * (sigmoid(X * beta) - Y)
    
    return grad

'''
@params: beta: np.array(float)
@post: calls to sigmoid(), np.log(); use of global variables X and Y
@returns: float
'''
def costFunctionOpt(beta):
    # the scipy optimize function passes beta as a list, as a result it must
    # be re-cast as a np.matrix in order for this algorithm to work
    beta = np.matrix(beta).transpose()
    
    # use the cost function provided where \hat(y) = sigmoid(X * beta)
    result = (-1.0 / X.shape[0]) * ((Y.transpose() * np.log(sigmoid(X*beta))) \
            + ((1 - Y).transpose() * np.log(1 - sigmoid(X*beta))))
    
    return result.item((0,0))

'''
@params: beta: np.matrix(float)
@post: calls to sigmoid(); use of global variable X
@returns: list(int)
'''
def predict(beta): 
    # initialize probabilities list to return
    prob = []

    # calculate the estimated Y given the beta coeffecients
    yhat = sigmoid(X * beta.transpose())

    # calculate conclusions for every row in the estimated Y
    for i in range(yhat.shape[0]):
        # if the sigmoid returns a probability of > 0.5 then conclude admitted
        if yhat[i][0] > 0.5:
            prob.append(1)
        else:
            prob.append(0)

    return prob

'''
@params: prediction: list(int); groundTruth: pd.DataFrame
@post: none
@returns: pd.DataFrame
'''
def confusionMatrix(prediction, groundTruth):
    # define the confusion matrix of type pd.DataFrame 
    conf_df = pd.crosstab(groundTruth, prediction, rownames=['True'], \
            colnames=['Predicted'], margins=False)
    return conf_df

# the ground truth used to compute accuracy
groundTruth = pd.Series(groundTruth)

# get and print results using gradient descent
grad_result = gradientDescent(beta, alpha=0.01, iters=1000)
grad_beta = grad_result[0].transpose()
print('Beta coeffecients calculated using Gradient Descent')
print(grad_beta)
print('Confusion Matrix using betas from Gradient Descent')
grad_pred = pd.Series(predict(grad_beta))
grad_conf = confusionMatrix(grad_pred, groundTruth)
print(grad_conf)
grad_conf_matrix = grad_conf.values
grad_acc = float(grad_conf_matrix.trace())/grad_conf_matrix.sum() * 100
print('Accuracy = ' + str(grad_acc) + '%\n')

# get and print results using the SciPy Optimization
opt_result = opt.fmin_tnc(func=costFunctionOpt, x0=beta, fprime=gradientOpt, \
        disp=False)
opt_beta = np.matrix(opt_result[0])
print('Beta coeffecients calculated using SciPy Optimization')
print(opt_beta)
print('Confusion Matrix using betas from SciPy Optimization')
opt_pred = pd.Series(predict(opt_beta))
opt_conf = confusionMatrix(opt_pred, groundTruth)
print(opt_conf)
opt_conf_matrix = opt_conf.values
opt_acc = float(opt_conf_matrix.trace())/opt_conf_matrix.sum() * 100
print('Accuracy = ' + str(opt_acc) + '%\n')

# TODO: DELETE BELOW
# Call the confusionMatrix function and print the confusion matrix as well as the accuracy of the model.
#The final outputs that we need for this portion of the lab are conf and acc. Copy conf and acc in a .txt file.
#Please write a SHORT report and explain these results. Include the explanations for both logistic and linear regression
#in the same PDF file. 
